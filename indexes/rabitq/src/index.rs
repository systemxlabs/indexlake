use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field};
use indexlake::expr::Expr;
use indexlake::index::{
    DynamicColumn, FilterIndexEntries, Index, RowValidity, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};
use uuid::Uuid;

use crate::RabitqMetric;
use crate::rabitq::{BruteForceRabitqIndex, BruteForceSearchParams};

#[derive(Debug)]
pub struct RabitqSearchQuery {
    pub vector: Vec<f32>,
    pub limit: usize,
}

impl RabitqSearchQuery {
    pub fn new(vector: Vec<f32>, limit: usize) -> Self {
        Self { vector, limit }
    }
}

impl SearchQuery for RabitqSearchQuery {
    fn index_kind(&self) -> &str {
        "rabitq"
    }

    fn limit(&self) -> Option<usize> {
        Some(self.limit)
    }
}

pub struct RabitqIndex {
    pub inner: BruteForceRabitqIndex,
    pub row_ids: Vec<Uuid>,
    pub metric: RabitqMetric,
}

impl std::fmt::Debug for RabitqIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RabitqIndex")
            .field("metric", &self.metric)
            .finish()
    }
}

#[async_trait::async_trait]
impl Index for RabitqIndex {
    async fn search(
        &self,
        query: &dyn SearchQuery,
        dynamic_fields: &[String],
        validity: &RowValidity,
    ) -> ILResult<SearchIndexEntries> {
        let query = query.downcast_ref::<RabitqSearchQuery>().ok_or_else(|| {
            ILError::index(format!(
                "RaBitQ index does not support search query: {query:?}"
            ))
        })?;

        let score_higher_is_better = matches!(self.metric, RabitqMetric::InnerProduct);

        // Over-fetch to account for invalid rows
        let fetch_limit = (query.limit * 2).max(query.limit);
        let params = BruteForceSearchParams { top_k: fetch_limit };
        let results = self
            .inner
            .search(&query.vector, params)
            .map_err(|e| ILError::index(format!("RaBitQ brute force search failed: {e}")))?;
        let mut row_ids = Vec::new();
        let mut scores = Vec::new();
        for r in results {
            if validity.is_valid(r.id) {
                row_ids.push(self.row_ids[r.id]);
                scores.push(r.score as f64);
                if row_ids.len() >= query.limit {
                    break;
                }
            }
        }

        let mut dynamic_columns = Vec::with_capacity(dynamic_fields.len());
        for dynamic_field in dynamic_fields {
            match dynamic_field.as_str() {
                "score" => dynamic_columns.push(DynamicColumn {
                    field: Arc::new(Field::new("score", DataType::Float64, false)),
                    values: Arc::new(Float64Array::from(scores.clone())),
                }),
                _ => {
                    return Err(ILError::invalid_input(format!(
                        "Unsupported dynamic field `{dynamic_field}` for rabitq index"
                    )));
                }
            }
        }

        Ok(SearchIndexEntries {
            row_ids,
            scores,
            score_higher_is_better,
            dynamic_columns,
        })
    }

    async fn filter(&self, _filters: &[Expr]) -> ILResult<FilterIndexEntries> {
        Err(ILError::not_supported(
            "RaBitQ index does not support filter",
        ))
    }
}
