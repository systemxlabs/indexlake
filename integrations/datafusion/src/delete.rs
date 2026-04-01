use std::sync::Arc;

use datafusion_common::DataFusionError;
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::StreamExt;

use crate::{LazyTable, make_count_schema, make_result_batch};

#[derive(Debug)]
pub struct IndexLakeDeleteExec {
    pub lazy_table: LazyTable,
    pub condition: indexlake::expr::Expr,
    cache: PlanProperties,
}

impl IndexLakeDeleteExec {
    pub fn try_new(
        lazy_table: LazyTable,
        condition: indexlake::expr::Expr,
    ) -> Result<Self, DataFusionError> {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(make_count_schema()),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );

        Ok(Self {
            lazy_table,
            condition,
            cache,
        })
    }
}

impl ExecutionPlan for IndexLakeDeleteExec {
    fn name(&self) -> &str {
        "IndexLakeDeleteExec"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn properties(&self) -> &PlanProperties {
        &self.cache
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
        if partition != 0 {
            return Err(DataFusionError::Execution(
                "IndexLakeDeleteExec can only be executed on a single partition".to_string(),
            ));
        }

        let lazy_table = self.lazy_table.clone();
        let condition = self.condition.clone();

        let stream = futures::stream::once(async move {
            let table = lazy_table
                .get_or_load()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let count = table
                .delete(condition)
                .await
                .map_err(|e| DataFusionError::Execution(e.to_string()))?;

            make_result_batch(count as i64)
        })
        .boxed();

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            make_count_schema(),
            stream,
        )))
    }
}

impl DisplayAs for IndexLakeDeleteExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "IndexLakeDeleteExec: table={}.{}",
            self.lazy_table.namespace_name, self.lazy_table.table_name
        )
    }
}
