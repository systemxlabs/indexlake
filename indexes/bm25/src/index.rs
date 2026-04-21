use std::sync::Arc;

use arrow::array::Float64Array;
use arrow::datatypes::{DataType, Field};
use bm25::Embedder;
use indexlake::expr::Expr;
use indexlake::index::{
    DynamicColumn, FilterIndexEntries, Index, IndexDefinitionRef, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};

use crate::{ArrowScorer, BM25IndexParams, JiebaTokenizer};

#[derive(Debug)]
pub struct BM25SearchQuery {
    pub query: String,
    pub limit: Option<usize>,
}

impl SearchQuery for BM25SearchQuery {
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
        dynamic_fields: &[String],
    ) -> ILResult<SearchIndexEntries> {
        let query = query
            .downcast_ref::<BM25SearchQuery>()
            .ok_or(ILError::index("Invalid query type"))?;
        let query_embedding = self.embedder.embed(&query.query);
        let matches = self.scorer.matches(&query_embedding, query.limit)?;

        let mut row_ids = Vec::with_capacity(matches.len());
        let mut scores = Vec::with_capacity(matches.len());
        for doc in matches {
            row_ids.push(doc.id);
            scores.push(doc.score as f64);
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
                        "Unsupported dynamic field `{dynamic_field}` for bm25 index"
                    )));
                }
            }
        }

        Ok(SearchIndexEntries {
            row_ids,
            scores,
            score_higher_is_better: true,
            dynamic_columns,
        })
    }

    async fn filter(&self, _filters: &[Expr]) -> ILResult<FilterIndexEntries> {
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
