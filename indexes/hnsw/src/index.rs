use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field};
use hnsw::Hnsw;
use indexlake::expr::Expr;
use indexlake::index::{
    DynamicColumn, FilterIndexEntries, Index, RowValidity, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};
use rand_pcg::Pcg64;
use space::Knn;
use uuid::Uuid;

use crate::Euclidean;

#[derive(Debug)]
pub struct HnswSearchQuery {
    pub vector: Vec<f32>,
    pub limit: usize,
}

impl SearchQuery for HnswSearchQuery {
    fn index_kind(&self) -> &str {
        "hnsw"
    }

    fn limit(&self) -> Option<usize> {
        Some(self.limit)
    }
}

pub struct HnswIndex {
    hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 24, 48>,
    row_ids: Vec<Uuid>,
}

impl HnswIndex {
    pub fn new(hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 24, 48>, row_ids: Vec<Uuid>) -> Self {
        Self { hnsw, row_ids }
    }
}

impl std::fmt::Debug for HnswIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndex").finish()
    }
}

#[async_trait::async_trait]
impl Index for HnswIndex {
    async fn search(
        &self,
        query: &dyn SearchQuery,
        dynamic_fields: &[String],
        validity: &RowValidity,
    ) -> ILResult<SearchIndexEntries> {
        let query = query.downcast_ref::<HnswSearchQuery>().ok_or_else(|| {
            ILError::index(format!(
                "Hnsw index does not support search query: {query:?}"
            ))
        })?;

        // Keep fetching until we get enough valid rows or exhaust the index
        let mut row_ids = Vec::new();
        let mut scores = Vec::new();
        let mut fetch_limit = query.limit;
        let mut scanned_count = 0;
        while row_ids.len() < query.limit && fetch_limit <= self.hnsw.len() {
            let neighbors = self.hnsw.knn(&query.vector, fetch_limit);
            for neighbor in &neighbors[scanned_count..] {
                if validity.is_valid(neighbor.index) {
                    row_ids.push(self.row_ids[neighbor.index]);
                    scores.push(neighbor.distance as f64);
                    if row_ids.len() >= query.limit {
                        break;
                    }
                }
            }
            scanned_count = neighbors.len();
            if fetch_limit >= self.hnsw.len() {
                break;
            }
            fetch_limit = (fetch_limit * 2).min(self.hnsw.len());
        }

        let mut dynamic_columns = Vec::with_capacity(dynamic_fields.len());
        for dynamic_field in dynamic_fields {
            match dynamic_field.as_str() {
                "distance" => dynamic_columns.push(DynamicColumn {
                    field: Arc::new(Field::new("distance", DataType::Float64, false)),
                    values: Arc::new(Float64Array::from(scores.clone())),
                }),
                _ => {
                    return Err(ILError::invalid_input(format!(
                        "Unsupported dynamic field `{dynamic_field}` for hnsw index"
                    )));
                }
            }
        }
        Ok(SearchIndexEntries {
            row_ids,
            scores,
            score_higher_is_better: false,
            dynamic_columns,
        })
    }

    async fn filter(&self, _filters: &[Expr]) -> ILResult<FilterIndexEntries> {
        Err(ILError::not_supported("Hnsw index does not support filter"))
    }
}
