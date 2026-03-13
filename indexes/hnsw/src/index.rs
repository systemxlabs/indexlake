use std::any::Any;
use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::Field;
use hnsw::Hnsw;
use indexlake::expr::Expr;
use indexlake::index::{
    FilterIndexEntries, Index, IndexResultColumn, IndexResultOptions, RowIdScore,
    SearchIndexEntries, SearchQuery,
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
    fn as_any(&self) -> &dyn Any {
        self
    }

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
        options: &IndexResultOptions,
    ) -> ILResult<SearchIndexEntries> {
        let query = query
            .as_any()
            .downcast_ref::<HnswSearchQuery>()
            .ok_or_else(|| {
                ILError::index(format!(
                    "Hnsw index does not support search query: {query:?}"
                ))
            })?;

        let limit = std::cmp::min(query.limit, self.hnsw.len());

        let neighbors = self.hnsw.knn(&query.vector, limit);
        let mut row_id_scores = vec![];
        for neighbor in neighbors {
            row_id_scores.push(RowIdScore {
                row_id: self.row_ids[neighbor.index],
                score: neighbor.distance as f64,
            });
        }

        let mut dynamic_columns = Vec::new();
        for column in &options.columns {
            match column.name.as_str() {
                "distance" => dynamic_columns.push(IndexResultColumn {
                    field: Arc::new(Field::new(
                        &column.output_name,
                        arrow::datatypes::DataType::Float64,
                        false,
                    )),
                    values: Arc::new(Float64Array::from(
                        row_id_scores
                            .iter()
                            .map(|row| row.score)
                            .collect::<Vec<_>>(),
                    )),
                }),
                _ => {
                    return Err(ILError::invalid_input(format!(
                        "Unsupported result column `{}` for index kind `hnsw`",
                        column.name
                    )));
                }
            }
        }

        Ok(SearchIndexEntries {
            row_id_scores,
            score_higher_is_better: false,
            dynamic_columns,
        })
    }

    async fn filter(
        &self,
        _filters: &[Expr],
        _options: &IndexResultOptions,
    ) -> ILResult<FilterIndexEntries> {
        Err(ILError::not_supported("Hnsw index does not support filter"))
    }
}
