use std::sync::Arc;

use datafusion::{
    arrow::{
        array::{ArrayRef, RecordBatch, UInt64Array},
        datatypes::{DataType, Field, Schema, SchemaRef},
    },
    common::stats::Precision,
    error::DataFusionError,
    execution::{SendableRecordBatchStream, TaskContext},
    logical_expr::dml::InsertOp,
    physical_expr::EquivalenceProperties,
    physical_plan::{
        DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
        PlanProperties, stream::RecordBatchStreamAdapter,
    },
};
use futures::StreamExt;
use indexlake::{
    catalog::INTERNAL_ROW_ID_FIELD_NAME,
    table::{Table, TableInsertion, check_and_rewrite_insert_batches},
    utils::schema_without_row_id,
};

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

        let input_schema = input.schema();
        let input_schema_without_row_id = Arc::new(schema_without_row_id(&input_schema));
        let empty_batch = RecordBatch::new_empty(input_schema_without_row_id);
        check_and_rewrite_insert_batches(&[empty_batch], &table.schema, &table.field_records)
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

            let mut count = 0u64;
            while let Some(batch) = input_stream.next().await {
                let mut batch = batch?;
                count += batch.num_rows() as u64;

                if let Ok(index) = batch.schema().index_of(INTERNAL_ROW_ID_FIELD_NAME) {
                    batch.remove_column(index);
                }

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

fn make_result_batch(count: u64) -> Result<RecordBatch, DataFusionError> {
    let schema = make_count_schema();
    let array = Arc::new(UInt64Array::from(vec![count])) as ArrayRef;
    let batch = RecordBatch::try_new(schema, vec![array])?;
    Ok(batch)
}

fn make_count_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "count",
        DataType::UInt64,
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
