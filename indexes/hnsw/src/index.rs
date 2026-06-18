use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field};
use hnsw::{Hnsw, Searcher};
use indexlake::expr::Expr;
use indexlake::index::{
    DynamicColumn, FilterIndexEntries, Index, RowValidity, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};
use rand_pcg::Pcg64;
use space::Neighbor;
use uuid::Uuid;

use crate::Euclidean;

fn default_ef_search() -> usize {
    100
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswSearchQuery {
    pub vector: Vec<f32>,
    #[serde(default = "default_ef_search")]
    pub ef_search: usize,
}

impl SearchQuery for HnswSearchQuery {
    fn index_kind(&self) -> &str {
        "hnsw"
    }
}

#[derive(Debug)]
pub struct HnswSearchQueryCodec;

impl indexlake::index::SearchQueryCodec for HnswSearchQueryCodec {
    fn encode(&self, query: &dyn SearchQuery) -> indexlake::ILResult<Vec<u8>> {
        let q = query.downcast_ref::<HnswSearchQuery>().ok_or_else(|| {
            indexlake::ILError::index("HnswSearchQueryCodec encode: query is not HnswSearchQuery")
        })?;
        serde_json::to_vec(q).map_err(|e| {
            indexlake::ILError::index(format!("Failed to encode HnswSearchQuery: {e}"))
        })
    }

    fn decode(&self, data: &[u8]) -> indexlake::ILResult<Arc<dyn SearchQuery>> {
        serde_json::from_slice::<HnswSearchQuery>(data)
            .map(|q| Arc::new(q) as Arc<dyn SearchQuery>)
            .map_err(|e| {
                indexlake::ILError::index(format!("Failed to decode HnswSearchQuery: {e}"))
            })
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
        limit: Option<usize>,
    ) -> ILResult<SearchIndexEntries> {
        let query = query.downcast_ref::<HnswSearchQuery>().ok_or_else(|| {
            ILError::index(format!(
                "Hnsw index does not support search query: {query:?}"
            ))
        })?;

        let limit = limit.unwrap_or(usize::MAX);
        // Keep fetching until we get enough valid rows or exhaust the index
        let mut row_ids = Vec::new();
        let mut scores = Vec::new();
        let total = self.hnsw.len();
        // Use explicit ef_search to control search beam width independently
        // from the number of requested results. hnsw::nearest() uses ef to
        // bound the candidate pool, giving better recall than knn() which
        // internally sets ef = num + 16.
        let ef = query.ef_search.max(limit).min(total);
        let mut searcher = Searcher::<u32>::default();
        let mut dest: Vec<Neighbor<u32>> = vec![
            Neighbor {
                index: !0,
                distance: 0
            };
            ef
        ];
        let mut fetch_limit = limit.min(total);
        let mut scanned_count = 0;
        while row_ids.len() < limit && fetch_limit <= total {
            let found = self
                .hnsw
                .nearest(&query.vector, ef, &mut searcher, &mut dest);
            for neighbor in &found[scanned_count..] {
                if validity.is_valid(neighbor.index)? {
                    row_ids.push(self.row_ids[neighbor.index]);
                    scores.push(f32::from_bits(neighbor.distance) as f64);
                    if row_ids.len() >= limit {
                        break;
                    }
                }
            }
            scanned_count = found.len();
            if fetch_limit >= total {
                break;
            }
            fetch_limit = (fetch_limit * 2).min(total);
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

    async fn filter(
        &self,
        _filters: &[Expr],
        _validity: &RowValidity,
    ) -> ILResult<FilterIndexEntries> {
        Err(ILError::not_supported("Hnsw index does not support filter"))
    }
}
