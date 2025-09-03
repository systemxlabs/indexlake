use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::common::{DFSchema, Statistics, project_schema};
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::display::ProjectSchemaDisplay;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::limit::LimitStream;
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::prelude::Expr;
use futures::TryStreamExt;
use indexlake::table::{Table, TableScan, TableScanPartition};
use log::error;

use crate::datafusion_expr_to_indexlake_expr;

#[derive(Debug)]
pub struct IndexLakeScanExec {
    pub table: Arc<Table>,
    pub partition_count: usize,
    pub concurrency: Option<usize>,
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub limit: Option<usize>,
    properties: PlanProperties,
}

impl IndexLakeScanExec {
    pub fn try_new(
        table: Arc<Table>,
        partition_count: usize,
        concurrency: Option<usize>,
        projection: Option<Vec<usize>>,
        filters: Vec<Expr>,
        limit: Option<usize>,
    ) -> Result<Self, DataFusionError> {
        let projected_schema = project_schema(&table.schema, projection.as_ref())?;
        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Ok(Self {
            table,
            partition_count,
            concurrency,
            projection,
            filters,
            limit,
            properties,
        })
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

        let df_schema = DFSchema::try_from(self.table.schema.clone())?;
        let il_filters = self
            .filters
            .iter()
            .map(|f| datafusion_expr_to_indexlake_expr(f, &df_schema))
            .collect::<Result<Vec<_>, _>>()?;

        let mut scan = TableScan::default()
            .with_projection(self.projection.clone())
            .with_filters(il_filters)
            .with_partition(TableScanPartition {
                partition_idx: partition,
                partition_count: self.partition_count,
            });

        if let Some(limit) = self.limit {
            scan.batch_size = limit;
            scan.concurrency = limit / 10000 + 1;
        }

        // Override auto concurrency
        if let Some(concurrency) = self.concurrency {
            scan.concurrency = concurrency;
        }

        let projected_schema = self.schema();
        let fut = get_batch_stream(
            self.table.clone(),
            projected_schema.clone(),
            scan,
            self.limit,
        );
        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            projected_schema,
            stream,
        )))
    }

    fn partition_statistics(
        &self,
        partition: Option<usize>,
    ) -> Result<Statistics, DataFusionError> {
        let scan_partition = match partition {
            Some(partition) => TableScanPartition {
                partition_idx: partition,
                partition_count: self.partition_count,
            },
            None => TableScanPartition::single_partition(),
        };

        let row_count_result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async { self.table.count(scan_partition).await })
        });
        match row_count_result {
            Ok(row_count) => Ok(Statistics {
                num_rows: Precision::Exact(row_count),
                total_byte_size: Precision::Absent,
                column_statistics: Statistics::unknown_column(&self.schema()),
            }),
            Err(e) => Err(DataFusionError::Plan(format!(
                "Error getting indexlake table {}.{} row count: {:?}",
                self.table.namespace_name, self.table.table_name, e
            ))),
        }
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        match IndexLakeScanExec::try_new(
            self.table.clone(),
            self.partition_count,
            self.concurrency,
            self.projection.clone(),
            self.filters.clone(),
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
            self.table.namespace_name, self.table.table_name, self.partition_count
        )?;
        let projected_schema = self.schema();
        if !schema_projection_equals(&projected_schema, &self.table.schema) {
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
        if let Some(concurrency) = self.concurrency {
            write!(f, ", concurrency={concurrency}")?;
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
    scan: TableScan,
    limit: Option<usize>,
) -> Result<SendableRecordBatchStream, DataFusionError> {
    let stream = table
        .scan(scan)
        .await
        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
    let stream = stream.map_err(|e| DataFusionError::Execution(e.to_string()));
    let stream = Box::pin(RecordBatchStreamAdapter::new(projected_schema, stream));
    let metrics = BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
    let limit_stream = LimitStream::new(stream, 0, limit, metrics);
    Ok(Box::pin(limit_stream))
}
