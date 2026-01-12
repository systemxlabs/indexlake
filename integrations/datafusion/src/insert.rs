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
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    Partitioning, PlanProperties,
};
use futures::{StreamExt, TryStreamExt};
use indexlake::table::{Table, TableInsertion};
use indexlake::{Client, ILError};
use tokio::sync::Mutex;

#[derive(Debug)]
pub struct IndexLakeInsertExec {
    pub client: Arc<Client>,
    pub namespace_name: String,
    pub table_name: String,
    pub table: Arc<Mutex<Option<Arc<Table>>>>,
    pub input: Arc<dyn ExecutionPlan>,
    pub insert_op: InsertOp,
    pub stream_insert_threshold: usize,
    cache: PlanProperties,
}

impl IndexLakeInsertExec {
    pub fn try_new(
        client: Arc<Client>,
        table: Arc<Table>,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
        stream_insert_threshold: usize,
    ) -> Result<Self, DataFusionError> {
        match insert_op {
            InsertOp::Append | InsertOp::Overwrite => {}
            InsertOp::Replace => {
                return Err(DataFusionError::NotImplemented(
                    "Replace is not supported for indexlake table".to_string(),
                ));
            }
        }

        let cache = PlanProperties::new(
            EquivalenceProperties::new(make_count_schema()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            client,
            namespace_name: table.namespace_name.clone(),
            table_name: table.table_name.clone(),
            table: Arc::new(Mutex::new(Some(table))),
            input,
            insert_op,
            stream_insert_threshold,
            cache,
        })
    }

    pub fn try_new_without_table(
        client: Arc<Client>,
        namespace_name: String,
        table_name: String,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
        stream_insert_threshold: usize,
    ) -> Result<Self, DataFusionError> {
        match insert_op {
            InsertOp::Append | InsertOp::Overwrite => {}
            InsertOp::Replace => {
                return Err(DataFusionError::NotImplemented(
                    "Replace is not supported for indexlake table".to_string(),
                ));
            }
        }

        let cache = PlanProperties::new(
            EquivalenceProperties::new(make_count_schema()),
            Partitioning::UnknownPartitioning(1),
            input.pipeline_behavior(),
            input.boundedness(),
        );

        Ok(Self {
            client,
            namespace_name,
            table_name,
            table: Arc::new(Mutex::new(None)),
            input,
            insert_op,
            stream_insert_threshold,
            cache,
        })
    }

    pub fn with_table(self, table: Arc<Table>) -> Self {
        Self {
            table: Arc::new(Mutex::new(Some(table))),
            ..self
        }
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

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let exec = IndexLakeInsertExec::try_new_without_table(
            self.client.clone(),
            self.namespace_name.clone(),
            self.table_name.clone(),
            children[0].clone(),
            self.insert_op,
            self.stream_insert_threshold,
        )?;
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream, DataFusionError> {
        if partition != 0 {
            return Err(DataFusionError::Execution(
                "IndexLakeInsertExec can only be executed on a single partition".to_string(),
            ));
        }

        let mut input_stream = self.input.execute(partition, context)?;
        let table_mutex = self.table.clone();
        let client = self.client.clone();
        let namespace_name = self.namespace_name.clone();
        let table_name = self.table_name.clone();
        let input = self.input.clone();
        let insert_op = self.insert_op;
        let stream_insert_threshold = self.stream_insert_threshold;

        let stream = futures::stream::once(async move {
            let table = get_or_load_table_inner(
                &table_mutex,
                &client,
                &namespace_name,
                &table_name,
            )
            .await?;

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

            match input.partition_statistics(None).map(|stat| stat.num_rows) {
                Ok(Precision::Exact(num_rows)) | Ok(Precision::Inexact(num_rows))
                    if num_rows > stream_insert_threshold =>
                {
                    let stream = input_stream
                        .map_err(|err| {
                            ILError::invalid_input(format!(
                                "Failed to get batch from stream: {err}"
                            ))
                        })
                        .boxed();
                    let count = table.stream_insert(stream).await.map_err(|e| {
                        DataFusionError::Execution(format!(
                            "Failed to stream insert into indexlake: {e}"
                        ))
                    })?;
                    make_result_batch(count as i64)
                }
                _ => {
                    let mut count = 0i64;
                    while let Some(batch) = input_stream.next().await {
                        let batch = batch?;
                        count += batch.num_rows() as i64;

                        table
                            .insert(TableInsertion::new(vec![batch]).with_try_dump(false))
                            .await
                            .map_err(|e| DataFusionError::Execution(e.to_string()))?;
                    }

                    // trigger dump
                    table
                        .insert(TableInsertion::new(vec![]).with_try_dump(true))
                        .await
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                    make_result_batch(count)
                }
            }
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

async fn get_or_load_table_inner(
    table_mutex: &Arc<Mutex<Option<Arc<Table>>>>,
    client: &Arc<Client>,
    namespace_name: &str,
    table_name: &str,
) -> Result<Arc<Table>, DataFusionError> {
    let mut guard = table_mutex.lock().await;
    if let Some(table) = guard.as_ref() {
        return Ok(table.clone());
    }
    let table = client
        .load_table(namespace_name, table_name)
        .await
        .map_err(|e| DataFusionError::Internal(e.to_string()))?;
    let table = Arc::new(table);
    *guard = Some(table.clone());
    Ok(table)
}

impl DisplayAs for IndexLakeInsertExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IndexLakeInsertExec: table={}.{}",
            self.namespace_name, self.table_name
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
