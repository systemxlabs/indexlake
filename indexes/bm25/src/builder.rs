use arrow::array::{ArrayRef, AsArray, FixedSizeBinaryArray};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use bm25::{Embedder, EmbedderBuilder};
use futures::StreamExt;
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use crate::{ArrowScorer, BM25Index, BM25IndexParams, JiebaTokenizer};

#[derive(Debug)]
pub struct Bm25IndexBuilder {
    index_def: IndexDefinitionRef,
    params: BM25IndexParams,
    embedder: Embedder<u32, JiebaTokenizer>,
    scorer: ArrowScorer,
}

impl Bm25IndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<BM25IndexParams>()?.clone();
        let embedder = new_embedder(&params);
        Ok(Self {
            index_def,
            params,
            embedder,
            scorer: ArrowScorer::default(),
        })
    }
}

#[async_trait::async_trait]
impl IndexBuilder for Bm25IndexBuilder {
    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn is_empty(&self) -> bool {
        self.scorer.is_empty()
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch.column(key_column_index);

        append_batch_to_scorer(&self.embedder, &row_id_array, key_column, &mut self.scorer)
    }

    async fn read_file(&mut self, input_file: Box<dyn InputFile>) -> ILResult<()> {
        let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
        let mut batch_stream = arrow_reader_builder.build()?;

        while let Some(batch) = batch_stream.next().await {
            let batch = batch?;
            self.scorer.merge_record_batch(&batch)?;
        }

        Ok(())
    }

    async fn write_file(&mut self, output_file: Box<dyn OutputFile>) -> ILResult<()> {
        let writer_properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(4096))
            .build();
        let mut arrow_writer = AsyncArrowWriter::try_new(
            output_file,
            ArrowScorer::schema().clone(),
            Some(writer_properties),
        )?;

        let batch = self.scorer.record_batch()?;
        arrow_writer.write(&batch).await?;
        arrow_writer.close().await?;

        Ok(())
    }

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()> {
        let stream_reader = arrow::ipc::reader::StreamReader::try_new(buf, None)?;
        for batch in stream_reader {
            let batch = batch?;
            self.scorer.merge_record_batch(&batch)?;
        }
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let mut stream_writer =
            arrow::ipc::writer::StreamWriter::try_new(buf, ArrowScorer::schema())?;
        let batch = self.scorer.record_batch()?;
        stream_writer.write(&batch)?;
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        let mut scorer = std::mem::take(&mut self.scorer);
        scorer.finalize();
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

fn append_batch_to_scorer(
    embedder: &Embedder<u32, JiebaTokenizer>,
    row_id_array: &FixedSizeBinaryArray,
    key_column: &ArrayRef,
    scorer: &mut ArrowScorer,
) -> ILResult<()> {
    match key_column.data_type() {
        DataType::Utf8 => {
            let utf8_array = key_column.as_string::<i32>();
            append_embeddings(embedder, row_id_array, utf8_array.iter(), scorer)?;
        }
        DataType::LargeUtf8 => {
            let large_utf8_array = key_column.as_string::<i64>();
            append_embeddings(embedder, row_id_array, large_utf8_array.iter(), scorer)?;
        }
        DataType::Utf8View => {
            let utf8_view_array = key_column.as_string_view();
            append_embeddings(embedder, row_id_array, utf8_view_array.iter(), scorer)?;
        }
        data_type => {
            return Err(ILError::not_supported(format!(
                "Unsupported data type to compute embeddings: {data_type}"
            )));
        }
    }
    Ok(())
}

fn append_embeddings<'a, I>(
    embedder: &Embedder<u32, JiebaTokenizer>,
    row_id_array: &FixedSizeBinaryArray,
    values: I,
    scorer: &mut ArrowScorer,
) -> ILResult<()>
where
    I: IntoIterator<Item = Option<&'a str>>,
{
    for (row_id, value) in row_id_array.iter().zip(values) {
        let row_id = Uuid::from_slice(row_id.expect("row id is null"))?;
        let embedding = value.map(|v| embedder.embed(v));
        scorer.insert(row_id, embedding.as_ref());
    }
    Ok(())
}
