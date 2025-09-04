use std::any::Any;

use bm25::Embedder;
use indexlake::expr::Expr;
use indexlake::index::{
    FilterIndexEntries, Index, IndexDefinitionRef, RowIdScore, SearchIndexEntries, SearchQuery,
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
    async fn search(&self, query: &dyn SearchQuery) -> ILResult<SearchIndexEntries> {
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

        Ok(SearchIndexEntries {
            row_id_scores,
            score_higher_is_better: true,
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
