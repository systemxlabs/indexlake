use std::sync::Arc;

use arrow::array::{Array, FixedSizeBinaryArray, Float32Array, ListArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use bytes::{Buf, BufMut, Bytes, BytesMut};
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use rabitq_rs::{BruteForceRabitqIndex, IvfRabitqIndex, Metric, RotatorType};
use uuid::Uuid;

use crate::{RabitqAlgo, RabitqIndex, RabitqIndexInner, RabitqIndexParams, RabitqMetric};

pub struct RabitqIndexBuilder {
    index_def: IndexDefinitionRef,
    params: RabitqIndexParams,
    row_id_vector: Vec<(FixedSizeBinaryArray, ListArray)>,
    loaded: Option<RabitqIndex>,
}

impl RabitqIndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<RabitqIndexParams>()?.clone();
        Ok(Self {
            index_def,
            params,
            row_id_vector: Vec::new(),
            loaded: None,
        })
    }

    fn index_schema(&self) -> ILResult<SchemaRef> {
        let key_fields = self.index_def.key_fields()?;
        let key_field = key_fields
            .first()
            .ok_or_else(|| ILError::index("No key field found"))?;
        Ok(Arc::new(Schema::new(vec![
            Field::new("row_id", DataType::FixedSizeBinary(16), false),
            Field::new("vector", key_field.data_type().clone(), true),
        ])))
    }

    fn build_index(&self) -> ILResult<RabitqIndex> {
        let mut row_ids: Vec<Uuid> = Vec::new();
        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for (row_id_array, key_column) in &self.row_id_vector {
            for (row_id_bytes, vector_arr) in row_id_array.iter().zip(key_column.iter()) {
                let row_id = Uuid::from_slice(row_id_bytes.expect("row id is null"))?;
                if let Some(arr) = vector_arr {
                    let float_arr = arr
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| ILError::index("Vector is not a float32 array"))?;
                    vectors.push(float_arr.values().to_vec());
                    row_ids.push(row_id);
                }
            }
        }

        let metric = match self.params.metric {
            RabitqMetric::L2 => Metric::L2,
            RabitqMetric::InnerProduct => Metric::InnerProduct,
        };
        let inner = match self.params.algo {
            RabitqAlgo::BruteForce => {
                let index = BruteForceRabitqIndex::train(
                    &vectors,
                    self.params.total_bits,
                    metric,
                    RotatorType::FhtKacRotator,
                    42,
                    true,
                )
                .map_err(|e| ILError::index(format!("RaBitQ brute force train failed: {e}")))?;
                RabitqIndexInner::BruteForce(index)
            }
            RabitqAlgo::Ivf => {
                let index = IvfRabitqIndex::train(
                    &vectors,
                    self.params.nlist,
                    self.params.total_bits,
                    metric,
                    RotatorType::FhtKacRotator,
                    42,
                    true,
                )
                .map_err(|e| ILError::index(format!("RaBitQ IVF train failed: {e}")))?;
                RabitqIndexInner::Ivf(index)
            }
        };
        Ok(RabitqIndex {
            inner,
            row_ids,
            metric: self.params.metric.clone(),
        })
    }
}

/// On-disk format: 1 byte algo tag + row_ids (bincode) + rabitq index bytes.
/// Tag: 0 = BruteForce, 1 = Ivf.
fn serialize_index(index: &RabitqIndex) -> ILResult<BytesMut> {
    let mut buf = BytesMut::new();
    let row_ids_bytes = bincode::serialize(&index.row_ids)
        .map_err(|e| ILError::index(format!("Failed to serialize row ids: {e}")))?;

    let tag = match &index.inner {
        RabitqIndexInner::BruteForce(_) => 0u8,
        RabitqIndexInner::Ivf(_) => 1u8,
    };
    buf.put_u8(tag);
    buf.put_u64_le(row_ids_bytes.len() as u64);
    buf.put_slice(&row_ids_bytes);

    match &index.inner {
        RabitqIndexInner::BruteForce(idx) => idx.save_to_writer(&mut (&mut buf).writer()),
        RabitqIndexInner::Ivf(idx) => idx.save_to_writer(&mut (&mut buf).writer()),
    }
    .map_err(|e| ILError::index(format!("RaBitQ save failed: {e}")))?;

    Ok(buf)
}

fn deserialize_index(mut buf: Bytes, metric: RabitqMetric) -> ILResult<RabitqIndex> {
    if buf.is_empty() {
        return Err(ILError::index("Empty RaBitQ index data"));
    }
    let tag = buf.get_u8();
    let row_ids_len = buf.get_u64_le() as usize;
    let row_ids: Vec<Uuid> = bincode::deserialize(&buf[..row_ids_len])
        .map_err(|e| ILError::index(format!("Failed to deserialize row ids: {e}")))?;
    buf.advance(row_ids_len);
    let mut reader = buf.reader();
    let inner = match tag {
        0 => RabitqIndexInner::BruteForce(
            BruteForceRabitqIndex::load_from_reader(&mut reader)
                .map_err(|e| ILError::index(format!("RaBitQ load failed: {e}")))?,
        ),
        1 => RabitqIndexInner::Ivf(
            IvfRabitqIndex::load_from_reader(&mut reader)
                .map_err(|e| ILError::index(format!("RaBitQ load failed: {e}")))?,
        ),
        _ => return Err(ILError::index(format!("Unknown RaBitQ index tag: {tag}"))),
    };
    Ok(RabitqIndex {
        inner,
        row_ids,
        metric,
    })
}

impl std::fmt::Debug for RabitqIndexBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RabitqIndexBuilder")
            .field("index_def", &self.index_def)
            .field("params", &self.params)
            .finish()
    }
}

#[async_trait::async_trait]
impl IndexBuilder for RabitqIndexBuilder {
    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn is_empty(&self) -> bool {
        self.row_id_vector.is_empty() && self.loaded.is_none()
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;
        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch
            .column(key_column_index)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| ILError::index("Key column is not a list array"))?;
        self.row_id_vector.push((row_id_array, key_column.clone()));
        Ok(())
    }

    async fn read_file(&mut self, mut input_file: Box<dyn InputFile>) -> ILResult<()> {
        let file_meta = input_file.metadata().await?;
        let data = input_file.read(0..file_meta.size).await?;
        let index = deserialize_index(data, self.params.metric.clone())?;
        self.loaded = Some(index);
        Ok(())
    }

    async fn write_file(&mut self, mut output_file: Box<dyn OutputFile>) -> ILResult<()> {
        let index = self.build_index()?;
        let data = serialize_index(&index)?;
        output_file.write(data.freeze()).await?;
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
        let schema = self.index_schema()?;
        let mut stream_writer = arrow::ipc::writer::StreamWriter::try_new(buf, schema.as_ref())?;
        for (row_id_array, key_column) in &self.row_id_vector {
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(row_id_array.clone()), Arc::new(key_column.clone())],
            )?;
            stream_writer.write(&batch)?;
        }
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        if let Some(index) = self.loaded.take() {
            return Ok(Box::new(index));
        }
        let index = self.build_index()?;
        Ok(Box::new(index))
    }
}
