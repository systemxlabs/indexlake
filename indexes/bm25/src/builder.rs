use std::sync::{Arc, LazyLock};

use arrow::array::{
    ArrayRef, AsArray, FixedSizeBinaryArray, ListArray, ListBuilder, PrimitiveBuilder,
};
use arrow::datatypes::{DataType, Field, FieldRef, Float32Type, Schema, SchemaRef, UInt32Type};
use arrow::record_batch::RecordBatch;
use bm25::{Embedder, EmbedderBuilder, Embedding};
use futures::StreamExt;
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::properties::WriterProperties;

use crate::{ArrowScorer, BM25Index, BM25IndexParams, JiebaTokenizer};

static BM25_INDEX_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::FixedSizeBinary(16), false),
        Field::new(
            "embedding_indices",
            DataType::List(BM25_INDEX_SCHEMA_EMBEDDING_INDICES_INNER_FIELD.clone()),
            true,
        ),
        Field::new(
            "embedding_values",
            DataType::List(BM25_INDEX_SCHEMA_EMBEDDING_VALUES_INNER_FIELD.clone()),
            true,
        ),
    ]))
});

static BM25_INDEX_SCHEMA_EMBEDDING_INDICES_INNER_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::UInt32, false)));

static BM25_INDEX_SCHEMA_EMBEDDING_VALUES_INNER_FIELD: LazyLock<FieldRef> =
    LazyLock::new(|| Arc::new(Field::new("item", DataType::Float32, false)));

#[derive(Debug)]
pub struct Bm25IndexBuilder {
    index_def: IndexDefinitionRef,
    params: BM25IndexParams,
    embedder: Embedder<u32, JiebaTokenizer>,
    embeddings: Vec<(FixedSizeBinaryArray, ListArray, ListArray)>,
}

impl Bm25IndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<BM25IndexParams>()?.clone();
        let embedder = new_embedder(&params);
        Ok(Self {
            index_def,
            params,
            embedder,
            embeddings: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl IndexBuilder for Bm25IndexBuilder {
    fn mergeable(&self) -> bool {
        true
    }

    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch.column(key_column_index);

        let embeddings = compute_embeddings(&self.embedder, key_column)?;
        let (indices_array, values_array) = embeddings_to_arrays(&embeddings)?;
        self.embeddings
            .push((row_id_array, indices_array, values_array));
        Ok(())
    }

    async fn read_file(&mut self, input_file: Box<dyn InputFile>) -> ILResult<()> {
        let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
        let mut batch_stream = arrow_reader_builder.build()?;

        while let Some(batch) = batch_stream.next().await {
            let batch = batch?;
            let row_id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or(ILError::index(
                    "Failed to downcast row id to FixedSizeBinaryArray",
                ))?;
            let indices_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ILError::index("Failed to downcast indices to ListArray"))?;
            let values_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ILError::index("Failed to downcast values to ListArray"))?;
            self.embeddings.push((
                row_id_array.clone(),
                indices_array.clone(),
                values_array.clone(),
            ));
        }

        Ok(())
    }

    async fn write_file(&mut self, output_file: Box<dyn OutputFile>) -> ILResult<()> {
        let writer_properties = WriterProperties::builder()
            .set_max_row_group_size(4096)
            .build();
        let mut arrow_writer = AsyncArrowWriter::try_new(
            output_file,
            BM25_INDEX_SCHEMA.clone(),
            Some(writer_properties),
        )?;

        for (row_id_array, indices_array, values_array) in self.embeddings.iter() {
            let batch = RecordBatch::try_new(
                BM25_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(row_id_array.clone()),
                    Arc::new(indices_array.clone()),
                    Arc::new(values_array.clone()),
                ],
            )?;
            arrow_writer.write(&batch).await?;
        }

        arrow_writer.close().await?;

        Ok(())
    }

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()> {
        let stream_reader = arrow::ipc::reader::StreamReader::try_new(buf, None)?;
        for batch in stream_reader {
            let batch = batch?;
            let row_id_array = batch
                .column(0)
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .ok_or(ILError::index(
                    "Failed to downcast row id to FixedSizeBinaryArray",
                ))?;
            let indices_array = batch
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ILError::index("Failed to downcast indices to ListArray"))?;
            let values_array = batch
                .column(2)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or(ILError::index("Failed to downcast values to ListArray"))?;
            self.embeddings.push((
                row_id_array.clone(),
                indices_array.clone(),
                values_array.clone(),
            ));
        }
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let mut stream_writer = arrow::ipc::writer::StreamWriter::try_new(buf, &BM25_INDEX_SCHEMA)?;
        for (row_id_array, indices_array, values_array) in self.embeddings.iter() {
            let batch = RecordBatch::try_new(
                BM25_INDEX_SCHEMA.clone(),
                vec![
                    Arc::new(row_id_array.clone()),
                    Arc::new(indices_array.clone()),
                    Arc::new(values_array.clone()),
                ],
            )?;
            stream_writer.write(&batch)?;
        }
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        let mut scorer = ArrowScorer::new();
        for (row_id_array, indices_array, values_array) in self.embeddings.iter_mut() {
            scorer.insert(
                row_id_array.clone(),
                indices_array.clone(),
                values_array.clone(),
            )?;
        }
        let embedder = new_embedder(&self.params);
        let index = BM25Index {
            index_def: self.index_def.clone(),
            params: self.params.clone(),
            embedder,
            scorer,
        };
        Ok(Box::new(index))
    }
}

fn new_embedder(params: &BM25IndexParams) -> Embedder<u32, JiebaTokenizer> {
    EmbedderBuilder::with_avgdl(params.avgdl)
        .tokenizer(JiebaTokenizer)
        .build()
}

fn compute_embeddings(
    embedder: &Embedder<u32, JiebaTokenizer>,
    key_column: &ArrayRef,
) -> ILResult<Vec<Option<Embedding>>> {
    let data_type = key_column.data_type();
    let mut embeddings = Vec::with_capacity(key_column.len());
    match data_type {
        DataType::Utf8 => {
            let utf8_array = key_column.as_string::<i32>();
            for value in utf8_array.iter() {
                match value {
                    Some(value) => {
                        let embedding = embedder.embed(value);
                        embeddings.push(Some(embedding));
                    }
                    None => {
                        embeddings.push(None);
                    }
                }
            }
        }
        DataType::LargeUtf8 => {
            let large_utf8_array = key_column.as_string::<i64>();
            for value in large_utf8_array.iter() {
                match value {
                    Some(value) => {
                        let embedding = embedder.embed(value);
                        embeddings.push(Some(embedding));
                    }
                    None => {
                        embeddings.push(None);
                    }
                }
            }
        }
        DataType::Utf8View => {
            let utf8_view_array = key_column.as_string_view();
            for value in utf8_view_array.iter() {
                match value {
                    Some(value) => {
                        let embedding = embedder.embed(value);
                        embeddings.push(Some(embedding));
                    }
                    None => {
                        embeddings.push(None);
                    }
                }
            }
        }
        _ => {
            return Err(ILError::not_supported(format!(
                "Unsupported data type to compute embeddings: {data_type}"
            )));
        }
    }
    Ok(embeddings)
}

fn embeddings_to_arrays(embeddings: &[Option<Embedding>]) -> ILResult<(ListArray, ListArray)> {
    let mut indices_builder = ListBuilder::new(PrimitiveBuilder::<UInt32Type>::new())
        .with_field(BM25_INDEX_SCHEMA_EMBEDDING_INDICES_INNER_FIELD.clone());
    let mut values_builder = ListBuilder::new(PrimitiveBuilder::<Float32Type>::new())
        .with_field(BM25_INDEX_SCHEMA_EMBEDDING_VALUES_INNER_FIELD.clone());
    for embedding_opt in embeddings.iter() {
        if let Some(embedding) = embedding_opt {
            let (indices, values): (Vec<Option<u32>>, Vec<Option<f32>>) = embedding
                .iter()
                .map(|te| (Some(te.index), Some(te.value)))
                .unzip();
            indices_builder.append_value(indices);
            values_builder.append_value(values);
        } else {
            indices_builder.append(false);
            values_builder.append(false);
        }
    }
    let indices_array = indices_builder.finish();
    let values_array = values_builder.finish();

    Ok((indices_array, values_array))
}
