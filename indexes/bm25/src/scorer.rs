use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock};

use arrow::array::{AsArray, FixedSizeBinaryBuilder, ListBuilder, PrimitiveBuilder};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Schema, SchemaRef, UInt32Type};
use arrow::record_batch::RecordBatch;
use bm25::{Embedding, ScoredDocument};
use indexlake::{ILError, ILResult};
use uuid::Uuid;

static SCORER_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("row_ids", DataType::List(ROW_IDS_FIELD.clone()), false),
        Field::new(
            "posting_token_indices",
            DataType::List(POSTING_TOKEN_INDICES_FIELD.clone()),
            false,
        ),
        Field::new(
            "posting_doc_ids",
            DataType::List(POSTING_DOC_IDS_FIELD.clone()),
            false,
        ),
        Field::new(
            "posting_values",
            DataType::List(POSTING_VALUES_FIELD.clone()),
            false,
        ),
        Field::new(
            "document_frequency_tokens",
            DataType::List(DOCUMENT_FREQUENCY_TOKENS_FIELD.clone()),
            false,
        ),
        Field::new(
            "document_frequency_values",
            DataType::List(DOCUMENT_FREQUENCY_VALUES_FIELD.clone()),
            false,
        ),
    ]))
});

static ROW_IDS_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::FixedSizeBinary(16), false)));

static POSTING_TOKEN_INDICES_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::UInt32, false)));

static POSTING_DOC_IDS_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::UInt32, false)));

static POSTING_VALUES_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::Float32, false)));

static DOCUMENT_FREQUENCY_TOKENS_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::UInt32, false)));

static DOCUMENT_FREQUENCY_VALUES_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::UInt32, false)));

struct ScorerData {
    posting_token_indices: Vec<u32>,
    posting_doc_ids: Vec<u32>,
    posting_values: Vec<f32>,
    document_frequency_tokens: Vec<u32>,
    document_frequency_values: Vec<u32>,
}

#[derive(Debug, Default)]
pub struct ArrowScorer {
    row_ids: Vec<Uuid>,
    postings: HashMap<u32, Vec<Posting>>,
    document_frequencies: HashMap<u32, u32>,
    inverse_document_frequencies: HashMap<u32, f32>,
}

#[derive(Debug)]
struct Posting {
    doc_id: u32,
    value: f32,
}

impl ArrowScorer {
    pub fn is_empty(&self) -> bool {
        self.row_ids.is_empty()
    }

    pub fn insert(&mut self, row_id: Uuid, embedding: Option<&Embedding<u32>>) {
        let doc_id = self.row_ids.len() as u32;
        self.row_ids.push(row_id);
        if let Some(embedding) = embedding {
            let mut seen = HashSet::with_capacity(embedding.len());
            for (&token_index, &value) in embedding.indices().zip(embedding.values()) {
                if seen.insert(token_index) {
                    self.postings
                        .entry(token_index)
                        .or_default()
                        .push(Posting { doc_id, value });
                    *self.document_frequencies.entry(token_index).or_default() += 1;
                }
            }
        }
    }

    pub fn finalize(&mut self) {
        let total_documents = self.row_ids.len();
        self.inverse_document_frequencies.clear();
        self.inverse_document_frequencies
            .reserve(self.document_frequencies.len());
        if total_documents == 0 {
            return;
        }

        for (&token_index, &token_frequency) in &self.document_frequencies {
            self.inverse_document_frequencies
                .insert(token_index, compute_idf(total_documents, token_frequency));
        }
    }

    pub fn schema() -> &'static SchemaRef {
        &SCORER_SCHEMA
    }

    pub fn record_batch(&self) -> ILResult<RecordBatch> {
        let mut row_ids_builder =
            ListBuilder::new(FixedSizeBinaryBuilder::new(16)).with_field(ROW_IDS_FIELD.clone());
        for row_id in &self.row_ids {
            row_ids_builder.values().append_value(row_id.as_bytes())?;
        }
        row_ids_builder.append(true);

        let data = self.data();

        let mut posting_token_indices_builder =
            ListBuilder::new(PrimitiveBuilder::<UInt32Type>::new())
                .with_field(POSTING_TOKEN_INDICES_FIELD.clone());
        posting_token_indices_builder
            .values()
            .append_slice(&data.posting_token_indices);
        posting_token_indices_builder.append(true);

        let mut posting_doc_ids_builder = ListBuilder::new(PrimitiveBuilder::<UInt32Type>::new())
            .with_field(POSTING_DOC_IDS_FIELD.clone());
        posting_doc_ids_builder
            .values()
            .append_slice(&data.posting_doc_ids);
        posting_doc_ids_builder.append(true);

        let mut posting_values_builder = ListBuilder::new(PrimitiveBuilder::<Float32Type>::new())
            .with_field(POSTING_VALUES_FIELD.clone());
        posting_values_builder
            .values()
            .append_slice(&data.posting_values);
        posting_values_builder.append(true);

        let mut document_frequency_tokens_builder =
            ListBuilder::new(PrimitiveBuilder::<UInt32Type>::new())
                .with_field(DOCUMENT_FREQUENCY_TOKENS_FIELD.clone());
        document_frequency_tokens_builder
            .values()
            .append_slice(&data.document_frequency_tokens);
        document_frequency_tokens_builder.append(true);

        let mut document_frequency_values_builder =
            ListBuilder::new(PrimitiveBuilder::<UInt32Type>::new())
                .with_field(DOCUMENT_FREQUENCY_VALUES_FIELD.clone());
        document_frequency_values_builder
            .values()
            .append_slice(&data.document_frequency_values);
        document_frequency_values_builder.append(true);

        Ok(RecordBatch::try_new(
            SCORER_SCHEMA.clone(),
            vec![
                Arc::new(row_ids_builder.finish()),
                Arc::new(posting_token_indices_builder.finish()),
                Arc::new(posting_doc_ids_builder.finish()),
                Arc::new(posting_values_builder.finish()),
                Arc::new(document_frequency_tokens_builder.finish()),
                Arc::new(document_frequency_values_builder.finish()),
            ],
        )?)
    }

    pub fn merge_record_batch(&mut self, batch: &RecordBatch) -> ILResult<()> {
        if batch.num_columns() != SCORER_SCHEMA.fields().len() {
            return Err(ILError::index(format!(
                "Invalid BM25 scorer batch: expected {} columns, got {}",
                SCORER_SCHEMA.fields().len(),
                batch.num_columns()
            )));
        }

        let row_ids_column = batch.column(0).as_list::<i32>();
        let posting_token_indices_column = batch.column(1).as_list::<i32>();
        let posting_doc_ids_column = batch.column(2).as_list::<i32>();
        let posting_values_column = batch.column(3).as_list::<i32>();
        let document_frequency_tokens_column = batch.column(4).as_list::<i32>();
        let document_frequency_values_column = batch.column(5).as_list::<i32>();

        for (
            row_ids,
            posting_token_indices,
            posting_doc_ids,
            posting_values,
            document_frequency_tokens,
            document_frequency_values,
        ) in itertools::izip!(
            row_ids_column.iter(),
            posting_token_indices_column.iter(),
            posting_doc_ids_column.iter(),
            posting_values_column.iter(),
            document_frequency_tokens_column.iter(),
            document_frequency_values_column.iter(),
        ) {
            let row_ids = row_ids.ok_or(ILError::index(
                "Invalid BM25 scorer batch: row ids should not be null",
            ))?;
            let posting_token_indices = posting_token_indices.ok_or(ILError::index(
                "Invalid BM25 scorer batch: posting token indices should not be null",
            ))?;
            let posting_doc_ids = posting_doc_ids.ok_or(ILError::index(
                "Invalid BM25 scorer batch: posting doc ids should not be null",
            ))?;
            let posting_values = posting_values.ok_or(ILError::index(
                "Invalid BM25 scorer batch: posting values should not be null",
            ))?;
            let document_frequency_tokens = document_frequency_tokens.ok_or(ILError::index(
                "Invalid BM25 scorer batch: document frequency tokens should not be null",
            ))?;
            let document_frequency_values = document_frequency_values.ok_or(ILError::index(
                "Invalid BM25 scorer batch: document frequency values should not be null",
            ))?;

            let row_ids = row_ids
                .as_fixed_size_binary()
                .iter()
                .map(|row_id| Uuid::from_slice(row_id.expect("row id is null")))
                .collect::<Result<Vec<Uuid>, uuid::Error>>()?;
            let posting_token_indices_arr = posting_token_indices.as_primitive::<UInt32Type>();
            let posting_doc_ids_arr = posting_doc_ids.as_primitive::<UInt32Type>();
            let posting_values_arr = posting_values.as_primitive::<Float32Type>();
            let document_frequency_tokens_arr =
                document_frequency_tokens.as_primitive::<UInt32Type>();
            let document_frequency_values_arr =
                document_frequency_values.as_primitive::<UInt32Type>();

            self.merge_data(
                &row_ids,
                posting_token_indices_arr.values(),
                posting_doc_ids_arr.values(),
                posting_values_arr.values(),
                document_frequency_tokens_arr.values(),
                document_frequency_values_arr.values(),
            )?;
        }

        Ok(())
    }

    pub fn matches(
        &self,
        query_embedding: &Embedding<u32>,
        limit: Option<usize>,
    ) -> ILResult<Vec<ScoredDocument<Uuid>>> {
        if self.row_ids.is_empty() || matches!(limit, Some(0)) {
            return Ok(Vec::new());
        }

        let total_documents = self.row_ids.len();
        let mut scores_by_doc_id = vec![0f32; total_documents];
        for token_index in query_embedding.indices() {
            let Some(postings) = self.postings.get(token_index) else {
                continue;
            };
            let idf = self
                .inverse_document_frequencies
                .get(token_index)
                .copied()
                .unwrap_or_else(|| {
                    let token_frequency = self
                        .document_frequencies
                        .get(token_index)
                        .copied()
                        .unwrap_or(postings.len() as u32);
                    compute_idf(total_documents, token_frequency)
                });
            for posting in postings {
                scores_by_doc_id[posting.doc_id as usize] += idf * posting.value;
            }
        }

        let mut scores = Vec::new();
        for (doc_id, &score) in scores_by_doc_id.iter().enumerate() {
            if score > 0.0 {
                scores.push(ScoredDocument {
                    id: self.row_ids[doc_id],
                    score,
                });
            }
        }

        if let Some(limit) = limit
            && scores.len() > limit
        {
            scores.select_nth_unstable_by(limit, |a, b| b.score.total_cmp(&a.score));
            scores.truncate(limit);
        }
        scores.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));

        Ok(scores)
    }

    fn data(&self) -> ScorerData {
        let total_postings = self.postings.values().map(Vec::len).sum();
        let mut data = ScorerData {
            posting_token_indices: Vec::with_capacity(total_postings),
            posting_doc_ids: Vec::with_capacity(total_postings),
            posting_values: Vec::with_capacity(total_postings),
            document_frequency_tokens: Vec::with_capacity(self.document_frequencies.len()),
            document_frequency_values: Vec::with_capacity(self.document_frequencies.len()),
        };

        let mut token_indices: Vec<u32> = self.postings.keys().copied().collect();
        token_indices.sort_unstable();
        for token_index in token_indices {
            data.document_frequency_tokens.push(token_index);
            data.document_frequency_values
                .push(self.document_frequencies[&token_index]);
            for posting in &self.postings[&token_index] {
                data.posting_token_indices.push(token_index);
                data.posting_doc_ids.push(posting.doc_id);
                data.posting_values.push(posting.value);
            }
        }

        data
    }

    fn merge_data(
        &mut self,
        row_ids: &[Uuid],
        posting_token_indices: &[u32],
        posting_doc_ids: &[u32],
        posting_values: &[f32],
        document_frequency_tokens: &[u32],
        document_frequency_values: &[u32],
    ) -> ILResult<()> {
        if posting_token_indices.len() != posting_doc_ids.len()
            || posting_token_indices.len() != posting_values.len()
        {
            return Err(ILError::index(
                "Invalid BM25 scorer batch: posting columns have mismatched lengths",
            ));
        }
        if document_frequency_tokens.len() != document_frequency_values.len() {
            return Err(ILError::index(
                "Invalid BM25 scorer batch: document frequency columns have mismatched lengths",
            ));
        }

        let local_row_count = row_ids.len() as u32;
        if let Some(&max_doc_id) = posting_doc_ids.iter().max()
            && max_doc_id >= local_row_count
        {
            return Err(ILError::index(format!(
                "Invalid BM25 scorer batch: doc id {max_doc_id} is out of range for {local_row_count} rows"
            )));
        }

        let doc_id_offset = self.row_ids.len() as u32;
        self.row_ids.extend_from_slice(row_ids);

        for (&token_index, &doc_id, &value) in
            itertools::izip!(posting_token_indices, posting_doc_ids, posting_values,)
        {
            self.postings.entry(token_index).or_default().push(Posting {
                doc_id: doc_id_offset + doc_id,
                value,
            });
        }

        for (&token_index, &document_frequency) in document_frequency_tokens
            .iter()
            .zip(document_frequency_values)
        {
            *self.document_frequencies.entry(token_index).or_default() += document_frequency;
        }

        Ok(())
    }
}

fn compute_idf(total_documents: usize, token_frequency: u32) -> f32 {
    let numerator = total_documents as f32 - token_frequency as f32 + 0.5;
    let denominator = token_frequency as f32 + 0.5;
    (1f32 + (numerator / denominator)).ln()
}
