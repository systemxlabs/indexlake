use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::{DataFusionError, project_schema};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
use datafusion_physical_plan::display::ProjectSchemaDisplay;
use datafusion_physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use futures::TryStreamExt;
use indexlake::index::SearchQuery;
use indexlake::table::TableSearch;

use crate::LazyTable;

#[derive(Debug)]
pub struct IndexLakeSearchExec {
    pub lazy_table: LazyTable,
    pub output_schema: SchemaRef,
    pub query: Arc<dyn SearchQuery>,
    pub dynamic_fields: Vec<String>,
    pub projection: Option<Vec<usize>>,
    properties: Arc<PlanProperties>,
}

impl IndexLakeSearchExec {
    pub fn try_new(
        lazy_table: LazyTable,
        output_schema: SchemaRef,
        query: Arc<dyn SearchQuery>,
        dynamic_fields: Vec<String>,
        projection: Option<Vec<usize>>,
    ) -> Result<Self, DataFusionError> {
        let projected_schema = project_schema(&output_schema, projection.as_ref())?;
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(projected_schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            lazy_table,
            output_schema,
            query,
            dynamic_fields,
            projection,
            properties,
        })
    }
}

impl ExecutionPlan for IndexLakeSearchExec {
    fn name(&self) -> &str {
        "IndexLakeSearchExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn properties(&self) -> &Arc<PlanProperties> {
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
        if partition != 0 {
            return Err(DataFusionError::Execution(format!(
                "partition index out of range: {partition} >= 1"
            )));
        }

        let lazy_table = self.lazy_table.clone();
        let query = self.query.clone();
        let dynamic_fields = self.dynamic_fields.clone();
        let projection = self.projection.clone();

        let fut = async move {
            let table = lazy_table
                .get_or_load()
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;

            let search = TableSearch {
                query,
                projection: projection.clone(),
                dynamic_fields: dynamic_fields.clone(),
            };

            let stream = table
                .search(search)
                .await
                .map_err(|e| DataFusionError::External(Box::new(e)))?;
            let stream = stream.map_err(|e| DataFusionError::External(Box::new(e)));
            Ok::<_, DataFusionError>(stream)
        };

        let stream = futures::stream::once(fut).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            stream,
        )))
    }

    fn fetch(&self) -> Option<usize> {
        self.query.limit()
    }

    fn with_fetch(&self, _limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        // Create a new search query with the requested limit
        // Note: we can't modify the query in place, so we return None
        None
    }
}

impl DisplayAs for IndexLakeSearchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "IndexLakeSearchExec: table={}.{}, kind={}",
            self.lazy_table.namespace_name, self.lazy_table.table_name, self.query.index_kind()
        )?;
        if !self.dynamic_fields.is_empty() {
            write!(
                f,
                ", dynamic_fields=[{}]",
                self.dynamic_fields.join(", ")
            )?;
        }
        let projected_schema = self.schema();
        if !schema_projection_equals(&projected_schema, &self.output_schema) {
            write!(
                f,
                ", projection={}",
                ProjectSchemaDisplay(&projected_schema)
            )?;
        }
        if let Some(limit) = self.query.limit() {
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
