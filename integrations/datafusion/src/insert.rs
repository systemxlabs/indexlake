use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::stats::Precision;
use datafusion::error::DataFusionError;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties,
};
use futures::StreamExt;
use indexlake::table::{Table, TableInsertion, check_and_rewrite_insert_batches};

#[derive(Debug)]
pub struct IndexLakeInsertExec {
    pub table: Arc<Table>,
    pub input: Arc<dyn ExecutionPlan>,
    pub insert_op: InsertOp,
    cache: PlanProperties,
}

impl IndexLakeInsertExec {
    pub fn try_new(
        table: Arc<Table>,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Self, DataFusionError> {
        match insert_op {
            InsertOp::Append | InsertOp::Overwrite => {}
            InsertOp::Replace => {
                return Err(DataFusionError::NotImplemented(
                    "Replace is not supported for indexlake table".to_string(),
                ));
            }
        }

        let empty_batch = RecordBatch::new_empty(input.schema());
        check_and_rewrite_insert_batches(
            &[empty_batch],
            table.schema.clone(),
            &table.field_records,
            true,
        )
        .map_err(|e| DataFusionError::Plan(e.to_string()))?;

        let cache = PlanProperties::new(
            EquivalenceProperties::new(make_count_schema()),
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            table,
            input,
            insert_op,
            cache,
        })
    }
}

impl ExecutionPlan for IndexLakeInsertExec {
    fn name(&self) -> &str {
        "IndexLakeInsertExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let exec =
            IndexLakeInsertExec::try_new(self.table.clone(), children[0].clone(), self.insert_op)?;
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        let mut input_stream = self.input.execute(partition, context)?;
        let table = self.table.clone();
        let insert_op = self.insert_op;

        let stream = futures::stream::once(async move {
            match insert_op {
                InsertOp::Append => {}
                InsertOp::Overwrite => {
                    table
                        .truncate()
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                }
                InsertOp::Replace => {
                    return Err(DataFusionError::Execution(
                        "Replace is not supported".to_string(),
                    ));
                }
            }

            let mut count = 0i64;
            while let Some(batch) = input_stream.next().await {
                let batch = batch?;
                count += batch.num_rows() as i64;

                table
                    .insert(TableInsertion::new(vec![batch]))
                    .await
                    .map_err(|e| DataFusionError::Execution(e.to_string()))?;
            }

            make_result_batch(count)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_count_schema(),
            stream,
        )))
    }
}

fn make_result_batch(count: i64) -> Result<RecordBatch, DataFusionError> {
    let schema = make_count_schema();
    let array = Arc::new(Int64Array::from(vec![count])) as ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![array])?;
    Ok(batch)
}

pub fn make_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::Int64,
        false,
    )]))
}

impl DisplayAs for IndexLakeInsertExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IndexLakeInsertExec: table={}.{}",
            self.table.namespace_name, self.table.table_name
        )?;
        if let Ok(stats) = self.input.partition_statistics(None) {
            match stats.num_rows {
                Precision::Exact(rows) => write!(f, ", rows={rows}")?,
                Precision::Inexact(rows) => write!(f, ", rows~={rows}")?,
                Precision::Absent => {}
            }
        }
        Ok(())
    }
}
