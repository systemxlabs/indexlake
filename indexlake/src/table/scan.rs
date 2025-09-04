use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use futures::{StreamExt, TryStreamExt};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, CatalogSchema, DataFileRecord, IndexFileRecord, InlineIndexRecord,
    rows_to_record_batch,
};
use crate::expr::{Expr, split_conjunction_filters};
use crate::index::{FilterSupport, IndexManager};
use crate::storage::read_data_file_by_record;
use crate::table::Table;
use crate::utils::{fixed_size_binary_array_to_uuids, project_schema};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, Clone, derive_with::With)]
pub struct TableScan {
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub batch_size: usize,
    pub partition: TableScanPartition,
    pub concurrency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableScanPartition {
    pub partition_idx: usize,
    pub partition_count: usize,
}

impl TableScanPartition {
    pub fn single_partition() -> Self {
        Self {
            partition_idx: 0,
            partition_count: 1,
        }
    }

    pub fn validate(&self) -> ILResult<()> {
        if self.partition_count == 0 {
            return Err(ILError::invalid_input(
                "Partition count must be greater than 0",
            ));
        }
        if self.partition_idx >= self.partition_count {
            return Err(ILError::invalid_input(format!(
                "Partition index out of range: {} >= {}",
                self.partition_idx, self.partition_count
            )));
        }
        Ok(())
    }

    pub fn offset_limit(&self, count: usize) -> (usize, usize) {
        let partition_size = std::cmp::max(count / self.partition_count, 1);
        let offset = std::cmp::min(self.partition_idx * partition_size, count);
        let limit = if self.partition_idx == self.partition_count - 1 {
            count - offset
        } else {
            partition_size
        };
        (offset, limit)
    }
}

impl TableScan {
    pub fn projected_schema(&self, table_schema: &SchemaRef) -> ILResult<SchemaRef> {
        if let Some(projection) = &self.projection {
            let projected_schema = table_schema.project(projection)?;
            Ok(Arc::new(projected_schema))
        } else {
            Ok(table_schema.clone())
        }
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
    let mut streams: Vec<RecordBatchStream> = if scan.partition.partition_idx == 0 {
        // Scan inline rows
        let projected_schema = Arc::new(project_schema(&table.schema, scan.projection.as_ref())?);
        let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);
        let row_stream = catalog_helper
            .scan_inline_rows(&table.table_id, &catalog_schema, None, &scan.filters)
            .await?;
        let inline_stream = Box::pin(row_stream.chunks(scan.batch_size).map(move |rows| {
            let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
            let batch = rows_to_record_batch(&projected_schema, &rows)?;
            Ok(batch)
        }));
        vec![inline_stream]
    } else {
        vec![]
    };

    let data_file_count = catalog_helper.count_data_files(&table.table_id).await?;
    let (offset, limit) = scan.partition.offset_limit(data_file_count as usize);

    // Scan data files
    let data_file_records = catalog_helper
        .get_partitioned_data_files(&table.table_id, offset, limit)
        .await?;

    let mut futs = Vec::with_capacity(data_file_records.len());
    for data_file_record in data_file_records {
        let storage = table.storage.clone();
        let table_schema = table.schema.clone();
        let projection = scan.projection.clone();
        let filters = scan.filters.clone();
        let batch_size = scan.batch_size;
        let fut = async move {
            read_data_file_by_record(
                &storage,
                &table_schema,
                &data_file_record,
                projection,
                filters,
                None,
                batch_size,
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
    let mut streams: Vec<RecordBatchStream> = if scan.partition.partition_idx == 0 {
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

    let data_file_count = catalog_helper.count_data_files(&table.table_id).await?;
    let (offset, limit) = scan.partition.offset_limit(data_file_count as usize);

    let data_file_records = catalog_helper
        .get_partitioned_data_files(&table.table_id, offset, limit)
        .await?;

    let mut futs = Vec::with_capacity(data_file_records.len());
    for data_file_record in data_file_records {
        let catalog_helper = catalog_helper.clone();
        let table = table.clone();
        let scan = scan.clone();
        let index_filter_assignment = index_filter_assignment.clone();
        let fut = async move {
            index_scan_data_file(
                &catalog_helper,
                &table,
                &scan,
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

    let mut intersected_row_ids =
        fixed_size_binary_array_to_uuids(&filter_index_entries_list[0].row_ids)?
            .into_iter()
            .collect::<HashSet<_>>();
    for filter_index_entries in filter_index_entries_list.iter().skip(1) {
        let set = fixed_size_binary_array_to_uuids(&filter_index_entries.row_ids)?
            .into_iter()
            .collect::<HashSet<_>>();
        intersected_row_ids = intersected_row_ids.intersection(&set).cloned().collect();
    }

    let projected_schema = Arc::new(project_schema(&table.schema, scan.projection.as_ref())?);
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);
    let row_stream = catalog_helper
        .scan_inline_rows(
            &table.table_id,
            &catalog_schema,
            Some(
                intersected_row_ids
                    .into_iter()
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
    scan: &TableScan,
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
        &scan.filters,
        &index_file_records_map,
        index_filter_assignment,
    )
    .await?;

    let left_filters = scan
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

    read_data_file_by_record(
        &table.storage,
        &table.schema,
        data_file_record,
        scan.projection.clone(),
        left_filters,
        Some(&row_ids),
        scan.batch_size,
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

        let input_file = table
            .storage
            .open_file(&index_file_record.relative_path)
            .await?;

        let mut index_builder = index_kind.builder(index_def)?;
        index_builder.read_file(input_file).await?;

        let index = index_builder.build()?;

        let filter_index_entries = index.filter(&filters).await?;
        filter_index_entries_list.push(filter_index_entries);
    }

    let mut intersected_row_ids =
        fixed_size_binary_array_to_uuids(&filter_index_entries_list[0].row_ids)?
            .into_iter()
            .collect::<HashSet<_>>();
    for filter_index_entries in filter_index_entries_list.iter().skip(1) {
        let set = fixed_size_binary_array_to_uuids(&filter_index_entries.row_ids)?
            .into_iter()
            .collect::<HashSet<_>>();
        intersected_row_ids = intersected_row_ids.intersection(&set).cloned().collect();
    }

    Ok(intersected_row_ids)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partition_offset_limit() {
        let partition = TableScanPartition {
            partition_idx: 0,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(1);
        assert_eq!(offset, 0);
        assert_eq!(limit, 1);
        let partition = TableScanPartition {
            partition_idx: 1,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(1);
        assert_eq!(offset, 1);
        assert_eq!(limit, 0);

        let partition = TableScanPartition {
            partition_idx: 0,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(3);
        assert_eq!(offset, 0);
        assert_eq!(limit, 1);
        let partition = TableScanPartition {
            partition_idx: 1,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(3);
        assert_eq!(offset, 1);
        assert_eq!(limit, 2);

        let partition = TableScanPartition {
            partition_idx: 0,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(4);
        assert_eq!(offset, 0);
        assert_eq!(limit, 2);
        let partition = TableScanPartition {
            partition_idx: 1,
            partition_count: 2,
        };
        let (offset, limit) = partition.offset_limit(4);
        assert_eq!(offset, 2);
        assert_eq!(limit, 2);
    }
}
