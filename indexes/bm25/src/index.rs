use std::any::Any;
use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::Field;
use bm25::Embedder;
use indexlake::expr::Expr;
use indexlake::index::{
    FilterIndexEntries, Index, IndexDefinitionRef, IndexResultColumn, IndexResultOptions,
    RowIdScore, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};

use crate::{ArrowScorer, BM25IndexParams, JiebaTokenizer};

#[derive(Debug)]
pub struct BM25SearchQuery {
    pub query: String,
    pub limit: Option<usize>,
}

impl SearchQuery for BM25SearchQuery {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn index_kind(&self) -> &str {
        "bm25"
    }

    fn limit(&self) -> Option<usize> {
        self.limit
    }
}

pub struct BM25Index {
    pub index_def: IndexDefinitionRef,
    pub params: BM25IndexParams,
    pub embedder: Embedder<u32, JiebaTokenizer>,
    pub scorer: ArrowScorer,
}

#[async_trait::async_trait]
impl Index for BM25Index {
    async fn search(
        &self,
        query: &dyn SearchQuery,
        options: &IndexResultOptions,
    ) -> ILResult<SearchIndexEntries> {
        let query = query
            .as_any()
            .downcast_ref::<BM25SearchQuery>()
            .ok_or(ILError::index("Invalid query type"))?;
        let query_embedding = self.embedder.embed(&query.query);
        let mut matches = self.scorer.matches(&query_embedding)?;
        if let Some(limit) = query.limit {
            matches.truncate(limit);
        }

        let row_id_scores: Vec<_> = matches
            .into_iter()
            .map(|doc| RowIdScore {
                row_id: doc.id,
                score: doc.score as f64,
            })
            .collect();

        let mut dynamic_columns = Vec::new();
        for column in &options.columns {
            match column.name.as_str() {
                "score" => dynamic_columns.push(IndexResultColumn {
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
                        "Unsupported result column `{}` for index kind `bm25`",
                        column.name
                    )));
                }
            }
        }

        Ok(SearchIndexEntries {
            row_id_scores,
            score_higher_is_better: true,
            dynamic_columns,
        })
    }

    async fn filter(
        &self,
        _filters: &[Expr],
        _options: &IndexResultOptions,
    ) -> ILResult<FilterIndexEntries> {
        Err(ILError::not_supported("BM25 index does not support filter"))
    }
}

impl std::fmt::Debug for BM25Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BM25Index")
            .field("index_def", &self.index_def)
            .field("params", &self.params)
            .field("embedder", &self.embedder)
            .finish()
    }
}
