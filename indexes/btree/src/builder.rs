use std::sync::Arc;

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use futures::StreamExt;
use indexlake::ILResult;
use indexlake::catalog::Scalar;
use indexlake::index::{Index, IndexBuilder, IndexDefinationRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::properties::WriterProperties;
use uuid::Uuid;

use crate::{BTreeIndex, OrderedScalar};

#[derive(Debug)]
pub struct BTreeIndexBuilder {
    index_def: IndexDefinationRef,
    index_schema: SchemaRef,
    index_batches: Vec<RecordBatch>,
}

impl BTreeIndexBuilder {
    pub fn try_new(index_def: IndexDefinationRef) -> ILResult<Self> {
        let key_column_name = &index_def.key_columns[0];
        let key_column_index = index_def.table_schema.index_of(key_column_name)?;
        let key_column_field = index_def.table_schema.fields[key_column_index].clone();
        let index_schema = Arc::new(Schema::new(vec![
            Arc::new(Field::new("row_id", DataType::FixedSizeBinary(16), false)),
            key_column_field,
        ]));

        Ok(Self {
            index_def,
            index_schema,
            index_batches: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl IndexBuilder for BTreeIndexBuilder {
    fn mergeable(&self) -> bool {
        true
    }

    fn index_def(&self) -> &IndexDefinationRef {
        &self.index_def
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch.column(key_column_index);

        let schema = self.index_schema.clone();
        let index_batch =
            RecordBatch::try_new(schema, vec![Arc::new(row_id_array), key_column.clone()])?;
        self.index_batches.push(index_batch);

        Ok(())
    }

    async fn read_file(&mut self, input_file: InputFile) -> ILResult<()> {
        let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
        let mut batch_stream = arrow_reader_builder.build()?;

        while let Some(batch) = batch_stream.next().await {
            let batch = batch?;
            self.index_batches.push(batch);
        }

        Ok(())
    }

    async fn write_file(&mut self, output_file: OutputFile) -> ILResult<()> {
        let writer_properties = WriterProperties::builder()
            .set_max_row_group_size(4096)
            .build();
        let mut arrow_writer = AsyncArrowWriter::try_new(
            output_file,
            self.index_schema.clone(),
            Some(writer_properties),
        )?;

        for batch in self.index_batches.iter() {
            arrow_writer.write(batch).await?;
        }

        arrow_writer.close().await?;

        Ok(())
    }

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()> {
        let stream_reader = StreamReader::try_new(buf, None)?;
        for batch in stream_reader {
            let batch = batch?;
            self.index_batches.push(batch);
        }
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let mut stream_writer = StreamWriter::try_new(buf, &self.index_schema)?;
        for batch in self.index_batches.iter() {
            stream_writer.write(batch)?;
        }
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        let mut index = BTreeIndex::new();

        for batch in self.index_batches.iter() {
            let row_id_array = batch.column(0).as_fixed_size_binary();
            let key_column = batch.column(1);

            for i in 0..batch.num_rows() {
                if key_column.is_valid(i) {
                    let key = OrderedScalar(Scalar::try_from_array(key_column, i)?);
                    let row_id = Uuid::from_slice(row_id_array.value(i))?;
                    index.insert(key, row_id)?;
                }
            }
        }

        Ok(Box::new(index))
    }
}
