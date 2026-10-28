use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field};
use indexlake::expr::Expr;
use indexlake::index::{
    DynamicColumn, FilterIndexEntries, Index, RowValidity, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};

use crate::graph::HnswGraph;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswSearchQuery {
    pub vector: Vec<f32>,
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

#[derive(Debug)]
pub struct HnswIndex {
    graph: HnswGraph,
}

impl HnswIndex {
    pub(crate) fn new(graph: HnswGraph) -> Self {
        Self { graph }
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
        let (row_ids, scores) = if self.graph.is_empty() || limit == 0 {
            (Vec::new(), Vec::new())
        } else {
            if query.vector.len() != self.graph.dim {
                return Err(ILError::invalid_input(format!(
                    "HNSW query dimension {} does not match index dimension {}",
                    query.vector.len(),
                    self.graph.dim
                )));
            }
            if query.vector.iter().any(|value| !value.is_finite()) {
                return Err(ILError::invalid_input(
                    "HNSW query vector must contain only finite values",
                ));
            }

            let node_count = self.graph.len();
            // A finite `limit` bounds the result set. When the caller does not
            // ask for a limit we have to visit every node to return all rows.
            let ef = if limit == usize::MAX {
                node_count
            } else {
                query.ef_search.max(limit).max(1).min(node_count)
            };
            let found = self.graph.search(&query.vector, ef, validity, limit)?;
            let mut row_ids = Vec::with_capacity(found.len());
            let mut scores = Vec::with_capacity(found.len());
            for (node_id, dist_sq) in found {
                row_ids.push(self.graph.row_ids[node_id as usize]);
                scores.push(dist_sq.max(0.0).sqrt() as f64);
            }
            (row_ids, scores)
        };

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
