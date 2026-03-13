use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use arrow_schema::{FieldRef, Schema};
use futures::{Stream, StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, CatalogSchema, DataFileRecord, IndexFileRecord, InlineIndexRecord,
    rows_to_record_batch,
};
use crate::expr::{Expr, merge_filters, row_ids_in_list_expr, split_conjunction_filters};
use crate::index::{
    FilterSupport, IndexColumnRequest, IndexManager, IndexResultOptions, RequestedIndexColumn,
};
use crate::storage::{Storage, count_data_file_by_record, read_data_file_by_record};
use crate::table::{Table, TableSchemaRef};
use crate::utils::{
    DynamicColumnLookup, append_columns_to_record_batch, extract_row_ids_from_record_batch,
    gather_index_result_columns, project_schema, validate_index_result_columns,
};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, Clone, derive_with::With)]
pub struct TableScan {
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub index_columns: Vec<IndexColumnRequest>,
    pub batch_size: usize,
    pub partition: TableScanPartition,
    pub offset: usize,
    pub limit: Option<usize>,
}

impl TableScan {
    pub fn validate(&self) -> ILResult<()> {
        if self.projection == Some(vec![]) {
            return Err(ILError::invalid_input(
                "projection must not be empty".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(ILError::invalid_input(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        self.partition.validate()?;
        Ok(())
    }

    pub fn output_schema(&self, table_schema: &Schema) -> ILResult<SchemaRef> {
        let projected_schema = if let Some(projection) = &self.projection {
            table_schema.project(projection)?
        } else {
            table_schema.clone()
        };
        Ok(Arc::new(projected_schema))
    }

    pub fn rewrite_columns(mut self, field_name_id_map: &HashMap<String, Uuid>) -> ILResult<Self> {
        let rewritten_filters = self
            .filters
            .into_iter()
            .map(|f| f.rewrite_columns(field_name_id_map))
            .collect::<ILResult<Vec<_>>>()?;
        self.filters = rewritten_filters;
        Ok(self)
    }

    pub fn offset_limit_required(&self) -> bool {
        self.offset > 0 || self.limit.is_some()
    }
}

impl Default for TableScan {
    fn default() -> Self {
        Self {
            projection: None,
            filters: vec![],
            index_columns: vec![],
            batch_size: 1024,
            partition: TableScanPartition::single_partition(),
            offset: 0,
            limit: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TableScanPartition {
    Auto {
        partition_idx: usize,
        partition_count: usize,
    },
    Provided {
        contains_inline_rows: bool,
        data_file_records: Vec<DataFileRecord>,
    },
}

impl TableScanPartition {
    pub fn single_partition() -> Self {
        Self::Auto {
            partition_idx: 0,
            partition_count: 1,
        }
    }

    pub fn data_file_offset_limit(
        data_file_count: usize,
        partition_count: usize,
        partition_idx: usize,
    ) -> (usize, usize) {
        if data_file_count == 0 {
            return (0, 0);
        }
        let partition_size = std::cmp::max(data_file_count / partition_count, 1);
        let offset = std::cmp::min(partition_idx * partition_size, data_file_count);
        let limit = if partition_idx == partition_count - 1 {
            data_file_count - offset
        } else {
            partition_size
        };
        (offset, limit)
    }

    pub fn validate(&self) -> ILResult<()> {
        match self {
            Self::Auto {
                partition_idx,
                partition_count,
            } => {
                if *partition_count == 0 {
                    return Err(ILError::invalid_input(
                        "Partition count must be greater than 0",
                    ));
                }
                if partition_idx >= partition_count {
                    return Err(ILError::invalid_input(format!(
                        "Partition index out of range: {partition_idx} >= {partition_count}",
                    )));
                }
            }
            Self::Provided { .. } => {}
        }
        Ok(())
    }

    pub fn contains_inline_rows(&self) -> bool {
        match self {
            Self::Auto { partition_idx, .. } => *partition_idx == 0,
            Self::Provided {
                contains_inline_rows,
                ..
            } => *contains_inline_rows,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct IndexDynamicResultPlan {
    options: IndexResultOptions,
    fields: Vec<FieldRef>,
}

#[derive(Debug, Clone, Default)]
struct ScanDynamicResultPlan {
    per_index: HashMap<String, IndexDynamicResultPlan>,
    output_fields: Vec<FieldRef>,
}

pub(crate) async fn process_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    mut scan: TableScan,
) -> ILResult<RecordBatchStream> {
    let filters = split_conjunction_filters(scan.filters.clone());
    scan.filters = filters;

    let index_filter_assignment = assign_index_filters(&table.index_manager, &scan.filters)?;
    let dynamic_result_plan =
        resolve_scan_dynamic_result_plan(table, &scan, &index_filter_assignment)?;

    if index_filter_assignment
        .values()
        .any(|filters| !filters.is_empty())
    {
        process_index_scan(
            catalog_helper,
            table,
            scan,
            index_filter_assignment,
            dynamic_result_plan,
        )
        .await
    } else if scan.index_columns.is_empty() {
        process_table_scan(catalog_helper, table, scan).await
    } else {
        Err(ILError::invalid_input(
            "index_columns requires at least one filter assigned to an index".to_string(),
        ))
    }
}

async fn process_table_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: TableScan,
) -> ILResult<RecordBatchStream> {
    let inline_row_count = if scan.offset_limit_required() && scan.partition.contains_inline_rows()
    {
        fast_count_inlint_rows(catalog_helper, &table.table_id, &table.table_schema, &scan).await?
    } else {
        None
    };

    let (inline_row_stream, inline_row_skip_count) = if scan.partition.contains_inline_rows() {
        // TODO we can skip stream if possible
        let inline_offset = if let Some(row_count) = inline_row_count {
            std::cmp::min(row_count, scan.offset)
        } else {
            0
        };
        let stream = scan_inline_rows(
            catalog_helper,
            &table.table_id,
            &table.table_schema,
            &scan,
            inline_offset,
        )
        .await?;
        (stream, inline_offset)
    } else {
        (
            Box::pin(futures::stream::iter(vec![])) as RecordBatchStream,
            0,
        )
    };

    let partitioned_data_file_records =
        get_partitioned_data_file_records(catalog_helper, &table.table_id, scan.partition.clone())
            .await?;

    let scanner = TablePartitionScanner::new(
        table.table_schema.clone(),
        table.storage.clone(),
        inline_row_stream,
        inline_row_skip_count,
        partitioned_data_file_records,
        scan,
    );

    Ok(Box::pin(scanner))
}

pub(crate) async fn get_partitioned_data_file_records(
    catalog_helper: &CatalogHelper,
    table_id: &Uuid,
    scan_partition: TableScanPartition,
) -> ILResult<Vec<DataFileRecord>> {
    match scan_partition {
        TableScanPartition::Auto {
            partition_idx,
            partition_count,
        } => {
            let data_file_count = catalog_helper.count_data_files(table_id).await? as usize;
            let (offset, limit) = TableScanPartition::data_file_offset_limit(
                data_file_count,
                partition_count,
                partition_idx,
            );
            if limit == 0 || offset >= data_file_count {
                return Ok(Vec::new());
            }

            // Scan data files
            let data_file_records = catalog_helper
                .get_partitioned_data_files(table_id, offset, limit)
                .await?;
            Ok(data_file_records)
        }
        TableScanPartition::Provided {
            data_file_records, ..
        } => Ok(data_file_records),
    }
}

async fn scan_inline_rows(
    catalog_helper: &CatalogHelper,
    table_id: &Uuid,
    table_schema: &TableSchemaRef,
    scan: &TableScan,
    offset: usize,
) -> ILResult<RecordBatchStream> {
    let projected_schema = Arc::new(project_schema(
        &table_schema.arrow_schema,
        scan.projection.as_ref(),
    )?);
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);

    let mut db_filters = Vec::new();
    let mut arrow_filters = Vec::new();
    for filter in scan.filters.iter() {
        if catalog_helper
            .catalog
            .supports_filter(filter, &table_schema.arrow_schema)?
        {
            db_filters.push(filter.clone());
        } else {
            arrow_filters.push(filter.clone());
        }
    }

    let arrow_filter = merge_filters(arrow_filters);

    let mut row_stream = catalog_helper
        .scan_inline_rows(
            table_id,
            &catalog_schema,
            None,
            &db_filters,
            if arrow_filter.is_some() {
                None
            } else {
                Some(offset)
            },
        )
        .await?;

    if arrow_filter.is_some() {
        row_stream = Box::pin(row_stream.skip(offset));
    }

    let inline_stream = Box::pin(row_stream.chunks(scan.batch_size).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let batch = rows_to_record_batch(&projected_schema, &rows)?;
        if let Some(arrow_filter) = &arrow_filter {
            let bool_array = arrow_filter.condition_eval(&batch)?;
            let filtered_batch = arrow::compute::filter_record_batch(&batch, &bool_array)?;
            Ok(filtered_batch)
        } else {
            Ok(batch)
        }
    }));
    Ok(Box::pin(inline_stream))
}

async fn fast_count_inlint_rows(
    catalog_helper: &CatalogHelper,
    table_id: &Uuid,
    table_schema: &TableSchemaRef,
    scan: &TableScan,
) -> ILResult<Option<usize>> {
    for filter in scan.filters.iter() {
        if !catalog_helper
            .catalog
            .supports_filter(filter, &table_schema.arrow_schema)?
        {
            return Ok(None);
        }
    }

    let count = catalog_helper
        .count_inline_rows(table_id, &scan.filters)
        .await?;
    Ok(Some(count as usize))
}

async fn process_index_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: TableScan,
    index_filter_assignment: HashMap<String, Vec<usize>>,
    dynamic_result_plan: ScanDynamicResultPlan,
) -> ILResult<RecordBatchStream> {
    let non_index_filters = scan
        .filters
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            !index_filter_assignment
                .values()
                .any(|indices| indices.contains(idx))
        })
        .map(|(_, filter)| filter.clone())
        .collect::<Vec<_>>();

    // Scan inline rows
    let mut streams: Vec<RecordBatchStream> = if scan.partition.contains_inline_rows() {
        let inline_rows_stream = index_scan_inline_rows(
            catalog_helper,
            table,
            &scan,
            &index_filter_assignment,
            &non_index_filters,
            &dynamic_result_plan,
        )
        .await?;
        vec![inline_rows_stream]
    } else {
        vec![]
    };

    let data_file_records =
        get_partitioned_data_file_records(catalog_helper, &table.table_id, scan.partition.clone())
            .await?;

    let mut futs = Vec::with_capacity(data_file_records.len());
    for data_file_record in data_file_records {
        let catalog_helper = catalog_helper.clone();
        let table = table.clone();
        let scan_projection = scan.projection.clone();
        let scan_filters = scan.filters.clone();
        let scan_batch_size = scan.batch_size;
        let index_filter_assignment = index_filter_assignment.clone();
        let dynamic_result_plan = dynamic_result_plan.clone();
        let fut = async move {
            index_scan_data_file(
                &catalog_helper,
                &table,
                scan_projection,
                &scan_filters,
                scan_batch_size,
                &data_file_record,
                &index_filter_assignment,
                &dynamic_result_plan,
            )
            .await
        };
        futs.push(fut);
    }
    let stream = futures::stream::iter(futs).buffered(1).try_flatten();

    streams.push(Box::pin(stream));
    let stream: RecordBatchStream = Box::pin(futures::stream::select_all(streams));
    if scan.offset_limit_required() {
        Ok(Box::pin(LimitOffsetRecordBatchStream::new(
            stream,
            scan.offset,
            scan.limit,
        )))
    } else {
        Ok(stream)
    }
}

async fn index_scan_inline_rows(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: &TableScan,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
    non_index_filters: &[Expr],
    dynamic_result_plan: &ScanDynamicResultPlan,
) -> ILResult<RecordBatchStream> {
    let mut index_builder_map = HashMap::new();
    let mut index_ids = Vec::new();
    for (index_name, _) in index_filter_assignment.iter() {
        let index_def = table
            .index_manager
            .get_index(index_name)
            .ok_or_else(|| ILError::internal(format!("Index {index_name} not found")))?;
        let kind = &index_def.kind;
        let index_kind = table
            .index_manager
            .get_index_kind(kind)
            .ok_or_else(|| ILError::internal(format!("Index kind {kind} not registered")))?;

        let index_builder = index_kind.builder(index_def)?;
        index_builder_map.insert(index_name, index_builder);
        index_ids.push(index_def.index_id);
    }

    // read inline index records
    let inline_index_records = catalog_helper.get_inline_indexes(&index_ids).await?;
    let mut inline_index_records_map: HashMap<Uuid, Vec<InlineIndexRecord>> = HashMap::new();
    for record in inline_index_records {
        inline_index_records_map
            .entry(record.index_id)
            .or_default()
            .push(record);
    }

    // append index builders
    for (_index_name, builder) in index_builder_map.iter_mut() {
        let index_def = builder.index_def();

        if let Some(records) = inline_index_records_map.get(&index_def.index_id) {
            for record in records {
                builder.read_bytes(&record.index_data)?;
            }
        }
    }

    // filter row ids by indexes
    let mut filter_index_entries_list = Vec::new();
    let mut dynamic_lookup_map = HashMap::new();
    for (index_name, filter_indices) in index_filter_assignment.iter() {
        let index_builder = index_builder_map.get_mut(index_name).ok_or_else(|| {
            ILError::internal(format!("Index builder not found for index {index_name}"))
        })?;

        let index = index_builder.build()?;

        let filters = filter_indices
            .iter()
            .map(|idx| scan.filters[*idx].clone())
            .collect::<Vec<_>>();

        let index_result_plan = dynamic_result_plan
            .per_index
            .get(index_name)
            .cloned()
            .unwrap_or_default();
        let filter_index_entries = index.filter(&filters, &index_result_plan.options).await?;
        validate_index_result_columns(
            &filter_index_entries.dynamic_columns,
            &index_result_plan.fields,
            filter_index_entries.row_ids.len(),
        )?;
        let row_ids = filter_index_entries.row_ids.clone();
        let lookups =
            build_dynamic_column_lookups(&row_ids, filter_index_entries.dynamic_columns.clone())?;
        dynamic_lookup_map.extend(
            lookups
                .into_iter()
                .map(|lookup| (lookup.field.name().clone(), lookup)),
        );
        filter_index_entries_list.push(filter_index_entries);
    }

    let mut intersected_row_ids = filter_index_entries_list[0]
        .row_ids
        .iter()
        .collect::<HashSet<_>>();
    for filter_index_entries in filter_index_entries_list.iter().skip(1) {
        let set = filter_index_entries.row_ids.iter().collect::<HashSet<_>>();
        intersected_row_ids = intersected_row_ids.intersection(&set).cloned().collect();
    }

    let projected_schema = Arc::new(project_schema(
        &table.table_schema.arrow_schema,
        scan.projection.as_ref(),
    )?);
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);
    let row_stream = catalog_helper
        .scan_inline_rows(
            &table.table_id,
            &catalog_schema,
            Some(
                intersected_row_ids
                    .into_iter()
                    .copied()
                    .collect::<Vec<_>>()
                    .as_slice(),
            ),
            non_index_filters,
            None,
        )
        .await?;
    let output_fields = dynamic_result_plan.output_fields.clone();
    let inline_stream = row_stream.chunks(scan.batch_size).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let batch = rows_to_record_batch(&projected_schema, &rows)?;
        append_requested_dynamic_columns(&batch, &dynamic_lookup_map, &output_fields)
    });

    Ok(Box::pin(inline_stream) as RecordBatchStream)
}

async fn index_scan_data_file(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan_projection: Option<Vec<usize>>,
    scan_filters: &[Expr],
    scan_batch_size: usize,
    data_file_record: &DataFileRecord,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
    dynamic_result_plan: &ScanDynamicResultPlan,
) -> ILResult<RecordBatchStream> {
    let index_file_records = catalog_helper
        .get_index_files_by_data_file_id(&data_file_record.data_file_id)
        .await?;
    let index_file_records_map = index_file_records
        .iter()
        .map(|record| (record.index_id, record))
        .collect::<HashMap<_, _>>();
    let index_filter_result = filter_index_files_row_ids(
        table,
        scan_filters,
        &index_file_records_map,
        index_filter_assignment,
        dynamic_result_plan,
    )
    .await?;

    let mut left_filters = scan_filters
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            !index_filter_assignment
                .values()
                .any(|indices| indices.contains(idx))
        })
        .map(|(_, filter)| filter.clone())
        .collect::<Vec<_>>();

    let row_id_filter = row_ids_in_list_expr(index_filter_result.row_ids.into_iter().collect());
    left_filters.push(row_id_filter);

    let stream = read_data_file_by_record(
        table.storage.as_ref(),
        &table.table_schema,
        data_file_record,
        scan_projection,
        left_filters,
        scan_batch_size,
    )
    .await?;
    if index_filter_result.lookups.is_empty() {
        Ok(stream)
    } else {
        let lookup_map = index_filter_result
            .lookups
            .into_iter()
            .map(|lookup| (lookup.field.name().clone(), lookup))
            .collect::<HashMap<_, _>>();
        let output_fields = dynamic_result_plan.output_fields.clone();
        let stream = stream.map(move |batch| {
            let batch = batch?;
            append_requested_dynamic_columns(&batch, &lookup_map, &output_fields)
        });
        Ok(Box::pin(stream))
    }
}

async fn filter_index_files_row_ids(
    table: &Table,
    filters: &[Expr],
    index_file_records: &HashMap<Uuid, &IndexFileRecord>,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
    dynamic_result_plan: &ScanDynamicResultPlan,
) -> ILResult<IndexFilterResult> {
    let mut filter_index_entries_list = Vec::new();
    let mut dynamic_lookup_map = HashMap::new();
    for (index_name, filter_indices) in index_filter_assignment.iter() {
        let index_def = table
            .index_manager
            .get_index(index_name)
            .ok_or_else(|| ILError::internal(format!("Index {index_name} not found")))?;
        let kind = &index_def.kind;
        let index_kind = table
            .index_manager
            .get_index_kind(kind)
            .ok_or_else(|| ILError::internal(format!("Index kind {kind} not registered")))?;

        let index_file_record = index_file_records.get(&index_def.index_id).ok_or_else(|| {
            ILError::internal(format!(
                "Index file record not found for index {index_name}"
            ))
        })?;

        let filters = filter_indices
            .iter()
            .map(|idx| filters[*idx].clone())
            .collect::<Vec<_>>();

        let input_file = table.storage.open(&index_file_record.relative_path).await?;

        let mut index_builder = index_kind.builder(index_def)?;
        index_builder.read_file(input_file).await?;

        let index = index_builder.build()?;

        let index_result_plan = dynamic_result_plan
            .per_index
            .get(index_name)
            .cloned()
            .unwrap_or_default();
        let filter_index_entries = index.filter(&filters, &index_result_plan.options).await?;
        validate_index_result_columns(
            &filter_index_entries.dynamic_columns,
            &index_result_plan.fields,
            filter_index_entries.row_ids.len(),
        )?;
        let row_ids = filter_index_entries.row_ids.clone();
        let lookups =
            build_dynamic_column_lookups(&row_ids, filter_index_entries.dynamic_columns.clone())?;
        dynamic_lookup_map.extend(
            lookups
                .into_iter()
                .map(|lookup| (lookup.field.name().clone(), lookup)),
        );
        filter_index_entries_list.push(filter_index_entries);
    }

    let mut intersected_row_ids = filter_index_entries_list[0]
        .row_ids
        .iter()
        .collect::<HashSet<_>>();
    for filter_index_entries in filter_index_entries_list.iter().skip(1) {
        let set = filter_index_entries.row_ids.iter().collect::<HashSet<_>>();
        intersected_row_ids = intersected_row_ids.intersection(&set).cloned().collect();
    }

    let lookups = dynamic_result_plan
        .output_fields
        .iter()
        .map(|field| {
            dynamic_lookup_map.remove(field.name()).ok_or_else(|| {
                ILError::internal(format!(
                    "Dynamic column lookup not found for {}",
                    field.name()
                ))
            })
        })
        .collect::<ILResult<Vec<_>>>()?;

    Ok(IndexFilterResult {
        row_ids: intersected_row_ids.into_iter().copied().collect(),
        lookups,
    })
}

fn assign_index_filters(
    index_manager: &IndexManager,
    filters: &[Expr],
) -> ILResult<HashMap<String, Vec<usize>>> {
    let mut index_filter_assignment: HashMap<String, Vec<usize>> = HashMap::new();
    for (filter_idx, filter) in filters.iter().enumerate() {
        let mut index_name = None;
        for (index_def, index_kind) in index_manager.iter_index_and_kind() {
            match index_kind.supports_filter(index_def, filter)? {
                // prioritize to assign filter to exact supported index
                FilterSupport::Exact => {
                    index_name = Some(index_def.name.clone());
                    break;
                }
                FilterSupport::Inexact => {
                    index_name = Some(index_def.name.clone());
                }
                FilterSupport::Unsupported => {}
            }
        }
        if let Some(index_name) = index_name {
            index_filter_assignment
                .entry(index_name)
                .or_default()
                .push(filter_idx);
        }
    }
    Ok(index_filter_assignment)
}

fn resolve_scan_dynamic_result_plan(
    table: &Table,
    scan: &TableScan,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
) -> ILResult<ScanDynamicResultPlan> {
    if scan.index_columns.is_empty() {
        return Ok(ScanDynamicResultPlan::default());
    }

    let participating_indexes = index_filter_assignment
        .iter()
        .filter(|(_, filter_indices)| !filter_indices.is_empty())
        .map(|(index_name, _)| index_name.clone())
        .collect::<Vec<_>>();
    if participating_indexes.is_empty() {
        return Err(ILError::invalid_input(
            "index_columns requires at least one filter assigned to an index".to_string(),
        ));
    }

    let projected_schema = project_schema(&table.output_schema, scan.projection.as_ref())?;
    let mut output_names = projected_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<HashSet<_>>();

    let mut grouped_columns: HashMap<String, Vec<RequestedIndexColumn>> = HashMap::new();
    let mut ordered_output_names = Vec::with_capacity(scan.index_columns.len());

    for column in &scan.index_columns {
        let target_index = if let Some(index_name) = &column.index_name {
            if !participating_indexes.contains(index_name) {
                return Err(ILError::invalid_input(format!(
                    "Scan result column `{}` targets index `{index_name}`, but that index does not participate in this scan",
                    column.name
                )));
            }
            index_name.clone()
        } else if participating_indexes.len() == 1 {
            participating_indexes[0].clone()
        } else {
            return Err(ILError::invalid_input(format!(
                "Scan result column `{}` must specify index_name because multiple indexes participate in this scan",
                column.name
            )));
        };

        let output_name = column.output_name().to_string();
        if !output_names.insert(output_name.clone()) {
            return Err(ILError::invalid_input(format!(
                "Duplicate output column name `{output_name}` in scan result"
            )));
        }

        grouped_columns
            .entry(target_index)
            .or_default()
            .push(RequestedIndexColumn {
                name: column.name.clone(),
                output_name,
            });
        ordered_output_names.push(column.output_name().to_string());
    }

    let mut per_index = HashMap::new();
    for (index_name, columns) in grouped_columns {
        let index_def = table
            .index_manager
            .get_index(&index_name)
            .ok_or_else(|| ILError::internal(format!("Index {index_name} not found")))?;
        let index_kind = table
            .index_manager
            .get_index_kind(&index_def.kind)
            .ok_or_else(|| {
                ILError::internal(format!("Index kind {} not registered", index_def.kind))
            })?;
        let fields = index_kind.output_fields(index_def.as_ref(), &columns)?;
        per_index.insert(
            index_name,
            IndexDynamicResultPlan {
                options: IndexResultOptions { columns },
                fields,
            },
        );
    }

    let output_fields = ordered_output_names
        .iter()
        .map(|name| {
            per_index
                .values()
                .flat_map(|plan| plan.fields.iter())
                .find(|field| field.name() == name)
                .cloned()
                .ok_or_else(|| {
                    ILError::internal(format!(
                        "Dynamic output field {name} not found after scan plan resolution"
                    ))
                })
        })
        .collect::<ILResult<Vec<_>>>()?;

    Ok(ScanDynamicResultPlan {
        per_index,
        output_fields,
    })
}

fn build_dynamic_column_lookups(
    row_ids: &[Uuid],
    dynamic_columns: Vec<crate::index::IndexResultColumn>,
) -> ILResult<Vec<DynamicColumnLookup>> {
    dynamic_columns
        .into_iter()
        .map(|column| DynamicColumnLookup::try_new(column.field, row_ids, column.values))
        .collect()
}

fn append_requested_dynamic_columns(
    batch: &RecordBatch,
    lookup_map: &HashMap<String, DynamicColumnLookup>,
    output_fields: &[FieldRef],
) -> ILResult<RecordBatch> {
    if output_fields.is_empty() {
        return Ok(batch.clone());
    }

    let row_ids = extract_row_ids_from_record_batch(batch)?;
    let lookups = output_fields
        .iter()
        .map(|field| {
            lookup_map.get(field.name()).cloned().ok_or_else(|| {
                ILError::internal(format!(
                    "Dynamic column lookup not found for {}",
                    field.name()
                ))
            })
        })
        .collect::<ILResult<Vec<_>>>()?;
    let columns = gather_index_result_columns(&row_ids, &lookups)?;
    append_columns_to_record_batch(batch, &columns)
}

#[derive(Debug)]
struct IndexFilterResult {
    row_ids: HashSet<Uuid>,
    lookups: Vec<DynamicColumnLookup>,
}

struct LimitOffsetRecordBatchStream {
    stream: RecordBatchStream,
    query_window: Range<usize>,
    row_pointer: usize,
}

impl LimitOffsetRecordBatchStream {
    fn new(stream: RecordBatchStream, offset: usize, limit: Option<usize>) -> Self {
        let query_window = if let Some(limit) = limit {
            offset..offset + limit
        } else {
            offset..usize::MAX
        };
        Self {
            stream,
            query_window,
            row_pointer: 0,
        }
    }

    fn apply_limit_offset(&self, batch: RecordBatch) -> RecordBatch {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return batch;
        }

        let batch_range = self.row_pointer..self.row_pointer + num_rows;
        let intersection_start = batch_range.start.max(self.query_window.start);
        let intersection_end = batch_range.end.min(self.query_window.end);
        if intersection_start >= intersection_end {
            return RecordBatch::new_empty(batch.schema());
        }

        let slice_start = intersection_start - batch_range.start;
        let slice_len = intersection_end - intersection_start;
        if slice_start == 0 && slice_len == num_rows {
            batch
        } else {
            batch.slice(slice_start, slice_len)
        }
    }
}

impl Stream for LimitOffsetRecordBatchStream {
    type Item = ILResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if self.row_pointer >= self.query_window.end {
                return Poll::Ready(None);
            }

            let stream = Pin::new(&mut self.stream);
            match stream.poll_next(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    let num_rows = batch.num_rows();
                    let batch = self.apply_limit_offset(batch);
                    self.row_pointer += num_rows;
                    if batch.num_rows() > 0 {
                        return Poll::Ready(Some(Ok(batch)));
                    }
                }
                Poll::Ready(Some(Err(err))) => return Poll::Ready(Some(Err(err))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

type GettingDataFileStreamFut =
    Pin<Box<dyn Future<Output = ILResult<GettingDataFileStreamResult>> + Send>>;

enum GettingDataFileStreamResult {
    Skip(usize),
    Streaming(RecordBatchStream),
}

enum ScanState {
    InlineRowStreaming(RecordBatchStream),
    GettingDataFileStream {
        idx: usize,
        fut: GettingDataFileStreamFut,
    },
    DataFileStreaming {
        idx: usize,
        stream: RecordBatchStream,
    },
    Done,
}

/// A scanner that streams record batches from both inline rows and data files.
pub struct TablePartitionScanner {
    table_schema: TableSchemaRef,
    storage: Arc<dyn Storage>,
    partitioned_data_file_records: Vec<DataFileRecord>,
    scan: TableScan,
    state: ScanState,
    query_window: Range<usize>,
    row_pointer: usize,
}

impl TablePartitionScanner {
    /// Create a new `TablePartitionScanner`.
    pub fn new(
        table_schema: TableSchemaRef,
        storage: Arc<dyn Storage>,
        inline_row_stream: RecordBatchStream,
        inline_row_skip_count: usize,
        partitioned_data_file_records: Vec<DataFileRecord>,
        scan: TableScan,
    ) -> Self {
        let state = ScanState::InlineRowStreaming(inline_row_stream);
        let row_pointer = inline_row_skip_count;

        // Create query range based on offset and limit
        let query_window = if let Some(limit) = scan.limit {
            scan.offset..scan.offset + limit
        } else {
            scan.offset..usize::MAX
        };

        Self {
            table_schema,
            storage,
            partitioned_data_file_records,
            scan,
            state,
            query_window,
            row_pointer,
        }
    }

    /// Check if the limit has been reached.
    fn is_limit_reached(&self) -> bool {
        self.row_pointer >= self.query_window.end
    }

    /// Apply offset and limit to a batch using Range operations.
    fn apply_limit_offset(&self, batch: RecordBatch) -> RecordBatch {
        let num_rows = batch.num_rows();
        if num_rows == 0 {
            return batch;
        }

        // Calculate the range of rows in this batch
        let batch_range = self.row_pointer..self.row_pointer + num_rows;

        // Find intersection with query window
        let intersection_start = batch_range.start.max(self.query_window.start);
        let intersection_end = batch_range.end.min(self.query_window.end);

        // Check if there's any overlap
        if intersection_start >= intersection_end {
            // No overlap, return empty batch
            return RecordBatch::new_empty(batch.schema());
        }

        // Calculate slice parameters relative to the batch
        let slice_start = intersection_start - batch_range.start;
        let slice_len = intersection_end - intersection_start;

        // Return slice if needed, otherwise return original batch
        if slice_start == 0 && slice_len == num_rows {
            // No slicing needed
            batch
        } else {
            batch.slice(slice_start, slice_len)
        }
    }

    #[allow(clippy::let_and_return)]
    fn get_stream_future(&self, record: DataFileRecord) -> GettingDataFileStreamFut {
        let fut = {
            let storage = Arc::clone(&self.storage);
            let table_schema = Arc::clone(&self.table_schema);
            let projection = self.scan.projection.clone();
            let filters = self.scan.filters.clone();
            let batch_size = self.scan.batch_size;
            let row_pointer = self.row_pointer;
            let query_window = self.query_window.clone();
            let needs_count = self.scan.offset_limit_required();

            Box::pin(async move {
                let count = if needs_count {
                    Some(
                        count_data_file_by_record(
                            storage.as_ref(),
                            &table_schema,
                            &record,
                            filters.clone(),
                        )
                        .await?,
                    )
                } else {
                    None
                };

                // Check if we can skip the entire file using range operations
                if let Some(file_count) = count {
                    let file_range = row_pointer..row_pointer + file_count;

                    // Check if file range has no intersection with query window
                    if file_range.end <= query_window.start || file_range.start >= query_window.end
                    {
                        // Skip entire file - no overlap with query window
                        return Ok(GettingDataFileStreamResult::Skip(file_count));
                    }
                }

                let stream = read_data_file_by_record(
                    storage.as_ref(),
                    &table_schema,
                    &record,
                    projection,
                    filters,
                    batch_size,
                )
                .await?;
                Ok(GettingDataFileStreamResult::Streaming(stream))
            })
        };
        fut
    }
}

impl Stream for TablePartitionScanner {
    type Item = ILResult<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let data_file_records_count = self.partitioned_data_file_records.len();

        loop {
            match &mut self.state {
                ScanState::InlineRowStreaming(stream) => {
                    let stream = Pin::new(stream);
                    match stream.poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            let num_rows = batch.num_rows();
                            let batch = self.apply_limit_offset(batch);
                            self.row_pointer += num_rows;
                            if batch.num_rows() > 0 {
                                return Poll::Ready(Some(Ok(batch)));
                            } else if self.is_limit_reached() {
                                self.state = ScanState::Done;
                                continue;
                            }
                        }
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                        Poll::Ready(None) => {
                            // Inline rows done, transition to Done or data files
                            if self.is_limit_reached() || data_file_records_count == 0 {
                                self.state = ScanState::Done;
                            } else {
                                let first_data_file_record =
                                    self.partitioned_data_file_records[0].clone();
                                let fut = self.get_stream_future(first_data_file_record);
                                self.state = ScanState::GettingDataFileStream { idx: 0, fut };
                            }
                            continue;
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ScanState::GettingDataFileStream { idx, fut } => {
                    let idx = *idx;
                    let mut fut = Pin::new(fut);
                    match fut.as_mut().poll(cx) {
                        Poll::Ready(Ok(result)) => {
                            match result {
                                GettingDataFileStreamResult::Skip(count) => {
                                    self.row_pointer += count;

                                    // Skip current file, move to next
                                    if self.is_limit_reached() || idx == data_file_records_count - 1
                                    {
                                        self.state = ScanState::Done;
                                    } else {
                                        let next_idx = idx + 1;
                                        let data_file_record =
                                            self.partitioned_data_file_records[next_idx].clone();
                                        let fut = self.get_stream_future(data_file_record);
                                        self.state =
                                            ScanState::GettingDataFileStream { idx: next_idx, fut };
                                    }
                                }
                                GettingDataFileStreamResult::Streaming(stream) => {
                                    self.state = ScanState::DataFileStreaming { idx, stream };
                                }
                            }
                            continue;
                        }
                        Poll::Ready(Err(e)) => return Poll::Ready(Some(Err(e))),
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ScanState::DataFileStreaming { idx, stream } => {
                    let idx = *idx;
                    let stream = Pin::new(stream);
                    match stream.poll_next(cx) {
                        Poll::Ready(Some(Ok(batch))) => {
                            let num_rows = batch.num_rows();
                            let batch = self.apply_limit_offset(batch);
                            self.row_pointer += num_rows;
                            if batch.num_rows() > 0 {
                                return Poll::Ready(Some(Ok(batch)));
                            } else if self.is_limit_reached() {
                                self.state = ScanState::Done;
                                continue;
                            }
                        }
                        Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                        Poll::Ready(None) => {
                            // Current file done, move to next
                            if self.is_limit_reached() || idx == data_file_records_count - 1 {
                                self.state = ScanState::Done;
                            } else {
                                let next_idx = idx + 1;
                                let data_file_record =
                                    self.partitioned_data_file_records[next_idx].clone();
                                let fut = self.get_stream_future(data_file_record);
                                self.state =
                                    ScanState::GettingDataFileStream { idx: next_idx, fut };
                            }
                            continue;
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                ScanState::Done => return Poll::Ready(None),
            }
        }
    }
}
