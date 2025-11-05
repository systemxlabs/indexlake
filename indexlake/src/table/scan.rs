use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use arrow::datatypes::SchemaRef;
use arrow_schema::{DataType, Field, FieldRef, Schema};
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, CatalogSchema, DataFileRecord, IndexFileRecord, InlineIndexRecord,
    inline_row_table_name, rows_to_record_batch,
};
use crate::expr::{Expr, merge_filters, split_conjunction_filters};
use crate::index::{FilterSupport, IndexManager};
use crate::storage::read_data_file_by_record;
use crate::table::{Table, TableSchemaRef};
use crate::utils::{
    append_new_fields_to_schema, project_schema, record_batch_with_location,
    record_batch_with_location_kind,
};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, Clone, derive_with::With)]
pub struct TableScan {
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub batch_size: usize,
    pub partition: TableScanPartition,
    pub concurrency: usize,
    pub metadata_columns: Vec<MetadataColumn>,
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
        if self.concurrency == 0 {
            return Err(ILError::invalid_input(
                "concurrency must be greater than 0".to_string(),
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
        let schema = append_new_fields_to_schema(
            &projected_schema,
            &self
                .metadata_columns
                .iter()
                .map(|column| column.to_field())
                .collect::<Vec<_>>(),
        );
        Ok(Arc::new(schema))
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
}

impl Default for TableScan {
    fn default() -> Self {
        Self {
            projection: None,
            filters: vec![],
            batch_size: 1024,
            partition: TableScanPartition::single_partition(),
            concurrency: num_cpus::get(),
            metadata_columns: vec![],
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum MetadataColumn {
    LocationKind,
    Location,
}

impl MetadataColumn {
    pub fn to_field(&self) -> FieldRef {
        static LOCATION_KIND_FIELD: LazyLock<FieldRef> = LazyLock::new(|| {
            Arc::new(Field::new(
                "_indexlake_location_kind",
                DataType::Utf8,
                false,
            ))
        });
        static LOCATION_FIELD: LazyLock<FieldRef> =
            LazyLock::new(|| Arc::new(Field::new("_indexlake_location", DataType::Utf8, false)));
        match self {
            MetadataColumn::LocationKind => LOCATION_KIND_FIELD.clone(),
            MetadataColumn::Location => LOCATION_FIELD.clone(),
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

pub(crate) async fn process_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    mut scan: TableScan,
) -> ILResult<RecordBatchStream> {
    let filters = split_conjunction_filters(scan.filters.clone());
    scan.filters = filters;

    let index_filter_assignment = assign_index_filters(&table.index_manager, &scan.filters)?;

    if index_filter_assignment
        .values()
        .any(|filters| !filters.is_empty())
    {
        process_index_scan(catalog_helper, table, scan, index_filter_assignment).await
    } else {
        process_table_scan(catalog_helper, table, scan).await
    }
}

async fn process_table_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: TableScan,
) -> ILResult<RecordBatchStream> {
    let mut streams: Vec<RecordBatchStream> = if scan.partition.contains_inline_rows() {
        // Scan inline rows
        let inline_stream = Box::pin(
            scan_inline_rows(catalog_helper, &table.table_id, &table.table_schema, &scan).await?,
        );
        vec![inline_stream]
    } else {
        vec![]
    };

    // Scan data files
    let data_file_records =
        get_partitioned_data_file_records(catalog_helper, &table.table_id, scan.partition).await?;

    let mut futs = Vec::with_capacity(data_file_records.len());
    for data_file_record in data_file_records {
        let storage = table.storage.clone();
        let table_schema = table.table_schema.clone();
        let projection = scan.projection.clone();
        let filters = scan.filters.clone();
        let batch_size = scan.batch_size;
        let metadata_columns = scan.metadata_columns.clone();
        let location = format!("data_file_{}", data_file_record.data_file_id);
        let fut = async move {
            let stream = read_data_file_by_record(
                storage.as_ref(),
                &table_schema,
                &data_file_record,
                projection,
                filters,
                None,
                batch_size,
            )
            .await?;
            let stream = stream.map(move |batch| {
                let mut batch = batch?;
                for meta_column in metadata_columns.iter() {
                    match meta_column {
                        MetadataColumn::LocationKind => {
                            batch = record_batch_with_location_kind(&batch, "data_file")?;
                        }
                        MetadataColumn::Location => {
                            batch = record_batch_with_location(&batch, location.as_str())?;
                        }
                    }
                }
                Ok::<_, ILError>(batch)
            });
            Ok::<_, ILError>(stream)
        };
        futs.push(fut);
    }
    let stream = futures::stream::iter(futs)
        .buffer_unordered(scan.concurrency)
        .try_flatten();

    streams.push(Box::pin(stream));

    Ok(Box::pin(futures::stream::select_all(streams)))
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

            let partition_size = std::cmp::max(data_file_count / partition_count, 1);
            let offset = std::cmp::min(partition_idx * partition_size, data_file_count);
            let limit = if partition_idx == partition_count - 1 {
                data_file_count - offset
            } else {
                partition_size
            };

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
    let metadata_columns = scan.metadata_columns.clone();
    let inline_row_table_name = inline_row_table_name(table_id);

    let row_stream = catalog_helper
        .scan_inline_rows(table_id, &catalog_schema, None, &db_filters)
        .await?;
    let inline_stream = Box::pin(row_stream.chunks(scan.batch_size).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let mut batch = rows_to_record_batch(&projected_schema, &rows)?;
        for meta_column in metadata_columns.iter() {
            match meta_column {
                MetadataColumn::LocationKind => {
                    batch = record_batch_with_location_kind(&batch, "inline")?;
                }
                MetadataColumn::Location => {
                    batch = record_batch_with_location(&batch, inline_row_table_name.as_str())?;
                }
            }
        }
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

async fn process_index_scan(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: TableScan,
    index_filter_assignment: HashMap<String, Vec<usize>>,
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
        )
        .await?;
        vec![inline_rows_stream]
    } else {
        vec![]
    };

    let data_file_records =
        get_partitioned_data_file_records(catalog_helper, &table.table_id, scan.partition).await?;

    let mut futs = Vec::with_capacity(data_file_records.len());
    for data_file_record in data_file_records {
        let catalog_helper = catalog_helper.clone();
        let table = table.clone();
        let scan_projection = scan.projection.clone();
        let scan_filters = scan.filters.clone();
        let scan_batch_size = scan.batch_size;
        let index_filter_assignment = index_filter_assignment.clone();
        let fut = async move {
            index_scan_data_file(
                &catalog_helper,
                &table,
                scan_projection,
                &scan_filters,
                scan_batch_size,
                &data_file_record,
                &index_filter_assignment,
            )
            .await
        };
        futs.push(fut);
    }
    let stream = futures::stream::iter(futs)
        .buffer_unordered(scan.concurrency)
        .try_flatten();

    streams.push(Box::pin(stream));

    Ok(Box::pin(futures::stream::select_all(streams)))
}

async fn index_scan_inline_rows(
    catalog_helper: &CatalogHelper,
    table: &Table,
    scan: &TableScan,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
    non_index_filters: &[Expr],
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
    for (index_name, builder) in index_builder_map.iter_mut() {
        let index_def = builder.index_def();

        if let Some(records) = inline_index_records_map.get(&index_def.index_id) {
            if records.len() > 1 && !builder.mergeable() {
                return Err(ILError::internal(format!(
                    "Index {index_name} is not mergeable but has multi inline index records"
                )));
            }
            for record in records {
                builder.read_bytes(&record.index_data)?;
            }
        }
    }

    // filter row ids by indexes
    let mut filter_index_entries_list = Vec::new();
    for (index_name, filter_indices) in index_filter_assignment.iter() {
        let index_builder = index_builder_map.get_mut(index_name).ok_or_else(|| {
            ILError::internal(format!("Index builder not found for index {index_name}"))
        })?;

        let index = index_builder.build()?;

        let filters = filter_indices
            .iter()
            .map(|idx| scan.filters[*idx].clone())
            .collect::<Vec<_>>();

        let filter_index_entries = index.filter(&filters).await?;
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
        )
        .await?;
    let inline_stream = row_stream.chunks(scan.batch_size).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let batch = rows_to_record_batch(&projected_schema, &rows)?;
        Ok::<_, ILError>(batch)
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
) -> ILResult<RecordBatchStream> {
    let index_file_records = catalog_helper
        .get_index_files_by_data_file_id(&data_file_record.data_file_id)
        .await?;
    let index_file_records_map = index_file_records
        .iter()
        .map(|record| (record.index_id, record))
        .collect::<HashMap<_, _>>();
    let row_ids = filter_index_files_row_ids(
        table,
        scan_filters,
        &index_file_records_map,
        index_filter_assignment,
    )
    .await?;

    let left_filters = scan_filters
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            !index_filter_assignment
                .values()
                .any(|indices| indices.contains(idx))
        })
        .map(|(_, filter)| filter.clone())
        .collect::<Vec<_>>();

    read_data_file_by_record(
        table.storage.as_ref(),
        &table.table_schema,
        data_file_record,
        scan_projection,
        left_filters,
        Some(row_ids.into_iter().collect()),
        scan_batch_size,
    )
    .await
}

async fn filter_index_files_row_ids(
    table: &Table,
    filters: &[Expr],
    index_file_records: &HashMap<Uuid, &IndexFileRecord>,
    index_filter_assignment: &HashMap<String, Vec<usize>>,
) -> ILResult<HashSet<Uuid>> {
    let mut filter_index_entries_list = Vec::new();
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

        let filter_index_entries = index.filter(&filters).await?;
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

    Ok(intersected_row_ids.into_iter().copied().collect())
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
