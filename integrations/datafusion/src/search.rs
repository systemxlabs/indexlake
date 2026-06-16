use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, project_schema};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::EquivalenceProperties;
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
        let exec_schema =
            merge_dynamic_fields(&lazy_table, &query, &projected_schema, &dynamic_fields);
        let properties = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(exec_schema),
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
            let table = lazy_table.get_or_load().await?;

            let search = TableSearch {
                query,
                projection: projection.clone(),
                dynamic_fields: dynamic_fields.clone(),
                concurrency: 8,
            };

            let stream = table.search(search).await?;
            let stream = stream.map_err(DataFusionError::from);
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
            self.lazy_table.namespace_name,
            self.lazy_table.table_name,
            self.query.index_kind()
        )?;
        if !self.dynamic_fields.is_empty() {
            write!(f, ", dynamic_fields=[{}]", self.dynamic_fields.join(", "))?;
        }
        if let Some(ref projection) = self.projection {
            write!(f, ", projection={projection:?}")?;
        }
        if let Some(limit) = self.query.limit() {
            write!(f, ", limit={limit}")?;
        }
        Ok(())
    }
}

/// Extend the projected schema with dynamic field columns.
///
/// Dynamic field Arrow types are resolved from the [`IndexKind`]
/// via `lazy_table.client.index_kinds`. Falls back to `DataType::Null`
/// placeholders if the index kind or its dynamic fields are not found.
fn merge_dynamic_fields(
    lazy_table: &LazyTable,
    query: &Arc<dyn SearchQuery>,
    projected_schema: &SchemaRef,
    dynamic_fields: &[String],
) -> SchemaRef {
    if dynamic_fields.is_empty() {
        return projected_schema.clone();
    }

    // Resolve dynamic field types from IndexKind
    let resolved_fields = lazy_table
        .client
        .index_kinds
        .get(query.index_kind())
        .and_then(|kind| kind.dynamic_fields().ok());

    let mut fields = projected_schema.fields().to_vec();
    if let Some(resolved) = resolved_fields {
        for field in resolved {
            if dynamic_fields.contains(&field.name().to_string()) {
                fields.push(field);
            }
        }
    } else {
        // Fallback: use Null placeholders
        for name in dynamic_fields {
            fields.push(Arc::new(arrow::datatypes::Field::new(
                name.clone(),
                arrow::datatypes::DataType::Null,
                true,
            )));
        }
    }
    Arc::new(arrow::datatypes::Schema::new_with_metadata(
        fields,
        projected_schema.metadata().clone(),
    ))
}
