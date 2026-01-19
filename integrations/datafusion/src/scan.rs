use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use datafusion::arrow::array::{RecordBatch, RecordBatchOptions};
use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::common::{DFSchema, Statistics, project_schema};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::display::ProjectSchemaDisplay;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::prelude::Expr;
use futures::{StreamExt, TryStreamExt};
use indexlake::catalog::DataFileRecord;
use indexlake::table::{Table, TableScan, TableScanPartition};
use log::error;

use crate::{LazyTable, datafusion_expr_to_indexlake_expr};

#[derive(Debug)]
pub struct IndexLakeScanExec {
    pub lazy_table: LazyTable,
    pub output_schema: SchemaRef,
    pub partition_count: usize,
    pub data_files: Option<Arc<Vec<DataFileRecord>>>,
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub batch_size: usize,
    pub limit: Option<usize>,
    pub data_file_partition_ranges: Option<Vec<Option<Range<usize>>>>,
    properties: PlanProperties,
}

impl IndexLakeScanExec {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        lazy_table: LazyTable,
        output_schema: SchemaRef,
        partition_count: usize,
        data_files: Option<Arc<Vec<DataFileRecord>>>,
        projection: Option<Vec<usize>>,
        filters: Vec<Expr>,
        batch_size: usize,
        limit: Option<usize>,
    ) -> Result<Self, DataFusionError> {
        let projected_schema = project_schema(&output_schema, projection.as_ref())?;
        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        let data_file_partition_ranges = data_files
            .as_ref()
            .map(|files| calc_data_file_partition_ranges(partition_count, files.len()));
        Ok(Self {
            lazy_table,
            output_schema,
            partition_count,
            data_files,
            projection,
            filters,
            batch_size,
            limit,
            data_file_partition_ranges,
            properties,
        })
    }

    pub fn get_scan_partition(&self, partition: Option<usize>) -> TableScanPartition {
        match partition {
            Some(partition) => {
                if let Some(data_files) = self.data_files.as_ref()
                    && let Some(data_file_partition_ranges) =
                        self.data_file_partition_ranges.as_ref()
                {
                    let range = data_file_partition_ranges[partition].clone();
                    TableScanPartition::Provided {
                        contains_inline_rows: partition == 0,
                        data_file_records: if let Some(range) = range {
                            data_files[range].to_vec()
                        } else {
                            vec![]
                        },
                    }
                } else {
                    TableScanPartition::Auto {
                        partition_idx: partition,
                        partition_count: self.partition_count,
                    }
                }
            }
            None => TableScanPartition::single_partition(),
        }
    }
}

impl ExecutionPlan for IndexLakeScanExec {
    fn name(&self) -> &str {
        "IndexLakeScanExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        if partition >= self.partition_count {
            return Err(DataFusionError::Execution(format!(
                "partition index out of range: {partition} >= {}",
                self.partition_count
            )));
        }

        let df_schema = DFSchema::try_from(self.output_schema.as_ref().clone())?;
        let il_filters = self
            .filters
            .iter()
            .map(|f| datafusion_expr_to_indexlake_expr(f, &df_schema))
            .collect::<Result<Vec<_>, _>>()?;

        let scan_partition = self.get_scan_partition(Some(partition));

        let scan = TableScan::default()
            .with_projection(self.projection.clone())
            .with_filters(il_filters)
            .with_batch_size(self.batch_size)
            .with_partition(scan_partition)
            .with_limit(self.limit);

        let projected_schema = self.schema();
        let lazy_table = self.lazy_table.clone();

        let fut = async move {
            let table = lazy_table
                .get_or_load()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            get_batch_stream(table, projected_schema.clone(), scan).await
        };
        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> Result<Statistics, DataFusionError> {
        let scan_partition = self.get_scan_partition(partition);
        let lazy_table = self.lazy_table.clone();

        let row_count_result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let table = lazy_table.get_or_load().await?;
                table.count(scan_partition).await
            })
        });
        match row_count_result {
            Ok(row_count) => {
                if self.filters.is_empty() {
                    if let Some(limit) = self.limit {
                        Ok(Statistics {
                            num_rows: Precision::Exact(std::cmp::min(row_count, limit)),
                            total_byte_size: Precision::Absent,
                            column_statistics: Statistics::unknown_column(&self.schema()),
                        })
                    } else {
                        Ok(Statistics {
                            num_rows: Precision::Exact(row_count),
                            total_byte_size: Precision::Absent,
                            column_statistics: Statistics::unknown_column(&self.schema()),
                        })
                    }
                } else {
                    Ok(Statistics {
                        num_rows: Precision::Inexact(row_count),
                        total_byte_size: Precision::Absent,
                        column_statistics: Statistics::unknown_column(&self.schema()),
                    })
                }
            }
            Err(e) => Err(DataFusionError::Plan(format!(
                "Error getting indexlake table {}.{} row count: {:?}",
                self.lazy_table.namespace_name, self.lazy_table.table_name, e
            ))),
        }
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        match IndexLakeScanExec::try_new(
            self.lazy_table.clone(),
            self.output_schema.clone(),
            self.partition_count,
            self.data_files.clone(),
            self.projection.clone(),
            self.filters.clone(),
            self.batch_size,
            limit,
        ) {
            Ok(exec) => Some(Arc::new(exec)),
            Err(e) => {
                error!("[indexlake] Failed to create IndexLakeScanExec with fetch: {e}");
                None
            }
        }
    }

    fn fetch(&self) -> Option<usize> {
        self.limit
    }
}

impl DisplayAs for IndexLakeScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "IndexLakeScanExec: table={}.{}, partitions={}",
            self.lazy_table.namespace_name, self.lazy_table.table_name, self.partition_count
        )?;
        let projected_schema = self.schema();
        if !schema_projection_equals(&projected_schema, &self.output_schema) {
            write!(
                f,
                ", projection={}",
                ProjectSchemaDisplay(&projected_schema)
            )?;
        }
        if !self.filters.is_empty() {
            write!(
                f,
                ", filters=[{}]",
                self.filters
                    .iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )?;
        }
        if let Some(limit) = self.limit {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }
}

fn schema_projection_equals(left: &Schema, right: &Schema) -> bool {
    if left.fields.len() != right.fields.len() {
        return false;
    }
    for (left_field, right_field) in left.fields.iter().zip(right.fields.iter()) {
        if left_field.name() != right_field.name() {
            return false;
        }
    }
    true
}

async fn get_batch_stream(
    table: Arc<Table>,
    projected_schema: SchemaRef,
    mut scan: TableScan,
) -> Result<SendableRecordBatchStream, DataFusionError> {
    let stream = if scan.projection == Some(Vec::new()) {
        scan.projection = Some(vec![0]);
        let stream = table
            .scan(scan)
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
        stream
            .map(|batch| {
                let batch = batch?;
                let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                let new_batch =
                    RecordBatch::try_new_with_options(Arc::new(Schema::empty()), vec![], &options)?;
                Ok(new_batch)
            })
            .boxed()
    } else {
        table
            .scan(scan)
            .await
            .map_err(|e| DataFusionError::Execution(e.to_string()))?
    };
    let stream = stream.map_err(|e| DataFusionError::Execution(e.to_string()));
    Ok(Box::pin(RecordBatchStreamAdapter::new(
        projected_schema,
        stream,
    )))
}

fn calc_data_file_partition_ranges(
    partition_count: usize,
    data_file_count: usize,
) -> Vec<Option<Range<usize>>> {
    let mut partition_allocations = vec![0; partition_count];

    if partition_count > data_file_count {
        for partition_allocation in partition_allocations.iter_mut().take(data_file_count) {
            *partition_allocation = 1;
        }
    } else {
        let partition_size = data_file_count / partition_count;
        for partition_allocation in partition_allocations.iter_mut() {
            *partition_allocation = partition_size;
        }

        let left = data_file_count - partition_count * partition_size;
        for partition_allocation in partition_allocations.iter_mut().take(left) {
            *partition_allocation += 1;
        }
    }

    let mut ranges = Vec::with_capacity(partition_count);
    let mut start = 0usize;
    for partition_allocation in partition_allocations.iter() {
        if *partition_allocation == 0 {
            ranges.push(None);
        } else {
            let partition_range = start..start + *partition_allocation;
            ranges.push(Some(partition_range));
            start += *partition_allocation;
        }
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_data_file_range() {
        let ranges = calc_data_file_partition_ranges(2, 0);
        assert_eq!(ranges, vec![None, None]);

        let ranges = calc_data_file_partition_ranges(2, 1);
        assert_eq!(ranges, vec![Some(0..1), None]);

        let ranges = calc_data_file_partition_ranges(2, 2);
        assert_eq!(ranges, vec![Some(0..1), Some(1..2)]);

        let ranges = calc_data_file_partition_ranges(2, 3);
        assert_eq!(ranges, vec![Some(0..2), Some(2..3)]);

        let ranges = calc_data_file_partition_ranges(2, 4);
        assert_eq!(ranges, vec![Some(0..2), Some(2..4)]);
    }
}
