use std::collections::{HashMap, HashSet};

use arrow::array::{Array, FixedSizeBinaryArray, Float32Array, ListArray, UInt32Array};
use bm25::{Embedding, ScoredDocument};
use indexlake::ILResult;
use uuid::Uuid;

#[derive(Default)]
pub struct ArrowScorer {
    embeddings: Vec<(FixedSizeBinaryArray, ListArray, ListArray)>,
    // A mapping from token indices to the set of documents that contain that token.
    inverted_token_index: HashMap<u32, HashSet<Uuid>>,
}

impl ArrowScorer {
    /// Creates a new `Scorer`.
    pub fn new() -> ArrowScorer {
        ArrowScorer {
            embeddings: Vec::new(),
            inverted_token_index: HashMap::new(),
        }
    }

    pub fn insert(
        &mut self,
        row_id: FixedSizeBinaryArray,
        indices: ListArray,
        values: ListArray,
    ) -> ILResult<()> {
        for (row_id_opt, index_array_opt) in row_id.iter().zip(indices.iter()) {
            let row_id = Uuid::from_slice(row_id_opt.expect("Row id should not be null"))?;
            if let Some(index_array) = index_array_opt {
                let indices = index_array
                    .as_any()
                    .downcast_ref::<UInt32Array>()
                    .expect("Indices should be UInt32Array");
                for token_index in indices.iter() {
                    let token_index = token_index.expect("Token index should not be null");
                    let documents_containing_token =
                        self.inverted_token_index.entry(token_index).or_default();
                    documents_containing_token.insert(row_id);
                }
            }
        }
        self.embeddings.push((row_id, indices, values));
        Ok(())
    }

    pub fn matches(&self, query_embedding: &Embedding<u32>) -> ILResult<Vec<ScoredDocument<Uuid>>> {
        let relevant_row_ids = query_embedding
            .indices()
            .filter_map(|token_index| self.inverted_token_index.get(token_index))
            .flat_map(|document_set| document_set.iter())
            .collect::<HashSet<_>>();

        let mut scores = self.build_scores(&relevant_row_ids, query_embedding)?;

        scores.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scores)
    }

    fn build_scores(
        &self,
        relevant_row_ids: &HashSet<&Uuid>,
        query_embedding: &Embedding<u32>,
    ) -> ILResult<Vec<ScoredDocument<Uuid>>> {
        let mut scores = Vec::new();
        for (row_id_array, indices_array, values_array) in self.embeddings.iter() {
            for ((row_id_opt, index_array_opt), value_array_opt) in row_id_array
                .iter()
                .zip(indices_array.iter())
                .zip(values_array.iter())
            {
                let row_id = Uuid::from_slice(row_id_opt.expect("Row id should not be null"))?;
                if relevant_row_ids.contains(&row_id)
                    && let Some(index_array) = index_array_opt
                    && let Some(value_array) = value_array_opt
                {
                    let index_array = index_array
                        .as_any()
                        .downcast_ref::<UInt32Array>()
                        .expect("Index array should be UInt32Array");
                    let value_array = value_array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .expect("Value array should be Float32Array");
                    let score = self.score_((index_array, value_array), query_embedding);
                    scores.push(ScoredDocument { id: row_id, score });
                }
            }
        }
        Ok(scores)
    }

    fn idf(&self, token_index: &u32) -> f32 {
        let token_frequency = self
            .inverted_token_index
            .get(token_index)
            .map_or(0, |documents| documents.len()) as f32;
        let mut total_documents = 0;
        for (row_id_array, ..) in self.embeddings.iter() {
            total_documents += row_id_array.len();
        }
        let numerator = total_documents as f32 - token_frequency + 0.5;
        let denominator = token_frequency + 0.5;
        (1f32 + (numerator / denominator)).ln()
    }

    fn score_(
        &self,
        document_embedding: (&UInt32Array, &Float32Array),
        query_embedding: &Embedding<u32>,
    ) -> f32 {
        let mut document_score = 0f32;

        for token_index in query_embedding.indices() {
            let token_idf = self.idf(token_index);
            let mut token_index_value = None;
            for (index, value) in document_embedding.0.iter().zip(document_embedding.1.iter()) {
                let index = index.expect("Index should not be null");
                let value = value.expect("Value should not be null");
                if index == *token_index {
                    token_index_value = Some(value);
                }
            }
            let token_score = token_idf * token_index_value.unwrap_or(0f32);
            document_score += token_score;
        }
        document_score
    }
}
