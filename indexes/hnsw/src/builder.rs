use arrow::array::{Float32Array, ListArray};
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
            hnsw,
            row_ids: vec![],
        })
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
    fn mergeable(&self) -> bool {
        false
    }

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

        for (row_id, vector_arr) in row_id_array.iter().zip(key_column.iter()) {
            let row_id = Uuid::from_slice(row_id.expect("row id is null"))?;
            if let Some(arr) = vector_arr {
                let vector = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| ILError::index("Vector is not a float32 array"))?;
                let vector = vector.values().to_vec();
                self.hnsw.insert(vector, &mut Searcher::default());
                self.row_ids.push(row_id);
            }
        }
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
        let hnsw_with_row_ids: HnswWithRowIds = bincode::deserialize(buf)
            .map_err(|e| ILError::index(format!("Failed to deserialize Hnsw and row ids: {e}")))?;
        self.hnsw = hnsw_with_row_ids.hnsw;
        self.row_ids = hnsw_with_row_ids.row_ids;
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let hnsw_with_row_ids = HnswWithRowIds {
            hnsw: std::mem::take(&mut self.hnsw),
            row_ids: std::mem::take(&mut self.row_ids),
        };
        let writer = std::io::Cursor::new(buf);
        bincode::serialize_into(writer, &hnsw_with_row_ids)
            .map_err(|e| ILError::index(format!("Failed to serialize Hnsw and row ids: {e}")))?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
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
