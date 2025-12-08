use std::sync::Arc;

use arrow::array::{FixedSizeBinaryArray, Float32Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use hnsw::{Hnsw, Params, Searcher};
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use rand_pcg::Pcg64;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{Euclidean, HnswIndex, HnswIndexParams};

pub struct HnswIndexBuilder {
    index_def: IndexDefinitionRef,
    params: HnswIndexParams,
    row_id_vector: Vec<(FixedSizeBinaryArray, ListArray)>,
    hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 24, 48>,
    row_ids: Vec<Uuid>,
}

impl HnswIndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<HnswIndexParams>()?.clone();

        let hnsw_params = Params::new().ef_construction(params.ef_construction);
        let hnsw = Hnsw::new_params(Euclidean, hnsw_params);

        Ok(Self {
            index_def,
            params,
            row_id_vector: Vec::new(),
            hnsw,
            row_ids: Vec::new(),
        })
    }

    pub fn index_schema(&self) -> ILResult<SchemaRef> {
        let key_fields = self.index_def.key_fields()?;
        let key_field = key_fields
            .first()
            .ok_or_else(|| ILError::index("No key field found"))?;
        Ok(Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::FixedSizeBinary(16), false),
            Field::new("vector", key_field.data_type().clone(), true),
        ])))
    }

    pub fn build_index(&mut self) -> ILResult<()> {
        let mut searcher = Searcher::default();
        for (row_id_array, key_column) in self.row_id_vector.iter() {
            for (row_id, vector_arr) in row_id_array.iter().zip(key_column.iter()) {
                let row_id = Uuid::from_slice(row_id.expect("row id is null"))?;
                if let Some(arr) = vector_arr {
                    let vector = arr
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| ILError::index("Vector is not a float32 array"))?;
                    let vector = vector.values().to_vec();
                    self.hnsw.insert(vector, &mut searcher);
                    self.row_ids.push(row_id);
                }
            }
        }
        Ok(())
    }
}

impl std::fmt::Debug for HnswIndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswIndexBuilder")
            .field("index_def", &self.index_def)
            .field("params", &self.params)
            .finish()
    }
}

#[async_trait::async_trait]
impl IndexBuilder for HnswIndexBuilder {
    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch.column(key_column_index);

        let key_column = key_column
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| ILError::index("Key column is not a list array"))?;

        self.row_id_vector.push((row_id_array, key_column.clone()));
        Ok(())
    }

    async fn read_file(&mut self, mut input_file: Box<dyn InputFile>) -> ILResult<()> {
        let file_meta = input_file.metadata().await?;
        let data = input_file.read(0..file_meta.size).await?;
        let hnsw_with_row_ids: HnswWithRowIds = bincode::deserialize(&data)
            .map_err(|e| ILError::index(format!("Failed to deserialize Hnsw and row ids: {e}")))?;
        self.hnsw = hnsw_with_row_ids.hnsw;
        self.row_ids = hnsw_with_row_ids.row_ids;
        Ok(())
    }

    async fn write_file(&mut self, mut output_file: Box<dyn OutputFile>) -> ILResult<()> {
        self.build_index()?;
        let hnsw_with_row_ids = HnswWithRowIds {
            hnsw: std::mem::take(&mut self.hnsw),
            row_ids: std::mem::take(&mut self.row_ids),
        };
        let data = bincode::serialize(&hnsw_with_row_ids)
            .map_err(|e| ILError::index(format!("Failed to serialize Hnsw and row ids: {e}")))?;
        output_file.write(Bytes::from_owner(data)).await?;
        output_file.close().await?;
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
                .ok_or_else(|| ILError::index("Row ID array is not a FixedSizeBinaryArray"))?;
            let key_column = batch
                .column(1)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| ILError::index("Key column is not a list array"))?;

            self.row_id_vector
                .push((row_id_array.clone(), key_column.clone()));
        }
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let mut stream_writer =
            arrow::ipc::writer::StreamWriter::try_new(buf, self.index_schema()?.as_ref())?;
        for (row_id_array, key_column) in self.row_id_vector.iter() {
            let batch = RecordBatch::try_new(
                self.index_schema()?,
                vec![Arc::new(row_id_array.clone()), Arc::new(key_column.clone())],
            )?;
            stream_writer.write(&batch)?;
        }
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        self.build_index()?;
        Ok(Box::new(HnswIndex::new(
            std::mem::take(&mut self.hnsw),
            std::mem::take(&mut self.row_ids),
        )))
    }
}

#[derive(Serialize, Deserialize)]
struct HnswWithRowIds {
    hnsw: Hnsw<Euclidean, Vec<f32>, Pcg64, 24, 48>,
    row_ids: Vec<Uuid>,
}
