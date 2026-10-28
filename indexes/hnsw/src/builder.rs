use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, ListArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use bytes::Bytes;
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use uuid::Uuid;

use crate::graph::HnswGraph;
use crate::serialization::{decode_graph, decode_pending, encode_graph, write_pending};
use crate::{HnswIndex, HnswIndexParams};

#[derive(Debug)]
pub struct HnswIndexBuilder {
    index_def: IndexDefinitionRef,
    params: HnswIndexParams,
    dim: Option<usize>,
    row_ids: Vec<Uuid>,
    present: Vec<bool>,
    vectors: Vec<f32>,
    graph: Option<HnswGraph>,
}

impl HnswIndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<HnswIndexParams>()?.clone();
        Ok(Self {
            index_def,
            params,
            dim: None,
            row_ids: Vec::new(),
            present: Vec::new(),
            vectors: Vec::new(),
            graph: None,
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

    fn push_absent(&mut self, row_id: Uuid) {
        self.row_ids.push(row_id);
        self.present.push(false);
    }

    fn push_vector(
        &mut self,
        row_id: Uuid,
        values: &Float32Array,
        offset: usize,
        width: usize,
    ) -> ILResult<()> {
        if width == 0 {
            return Err(ILError::index("HNSW vectors must not be empty"));
        }
        match self.dim {
            Some(dim) if dim != width => {
                return Err(ILError::index(format!(
                    "HNSW vector dimension changed from {dim} to {width}"
                )));
            }
            None => self.dim = Some(width),
            _ => {}
        }

        let end = offset
            .checked_add(width)
            .filter(|&end| end <= values.len())
            .ok_or_else(|| ILError::index("HNSW vector offset out of bounds"))?;
        if (offset..end).any(|index| values.is_null(index)) {
            return Err(ILError::index("HNSW vector elements must not be null"));
        }
        let vector = &values.values()[offset..end];
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(ILError::index(
                "HNSW vectors must contain only finite values",
            ));
        }

        self.vectors.extend_from_slice(vector);
        self.row_ids.push(row_id);
        self.present.push(true);
        Ok(())
    }

    fn append_list_rows(
        &mut self,
        row_id_array: &FixedSizeBinaryArray,
        list: &ListArray,
    ) -> ILResult<()> {
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| ILError::index("HNSW vector values must be float32"))?;
        let offsets = list.value_offsets();
        for index in 0..list.len() {
            let row_id = row_id_at(row_id_array, index)?;
            if list.is_null(index) {
                self.push_absent(row_id);
                continue;
            }
            let start = usize::try_from(offsets[index])
                .map_err(|_| ILError::index("HNSW list offset must be non-negative"))?;
            let end = usize::try_from(offsets[index + 1])
                .map_err(|_| ILError::index("HNSW list offset must be non-negative"))?;
            if end < start {
                return Err(ILError::index("HNSW list offsets are not monotonic"));
            }
            self.push_vector(row_id, values, start, end - start)?;
        }
        Ok(())
    }

    fn append_fixed_size_list_rows(
        &mut self,
        row_id_array: &FixedSizeBinaryArray,
        list: &FixedSizeListArray,
    ) -> ILResult<()> {
        let values = list
            .values()
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| ILError::index("HNSW vector values must be float32"))?;
        let width = list.value_length() as usize;
        for index in 0..list.len() {
            let row_id = row_id_at(row_id_array, index)?;
            if list.is_null(index) {
                self.push_absent(row_id);
                continue;
            }
            let offset = usize::try_from(list.value_offset(index))
                .map_err(|_| ILError::index("HNSW list offset must be non-negative"))?;
            self.push_vector(row_id, values, offset, width)?;
        }
        Ok(())
    }

    fn merge_pending(
        &mut self,
        dim: Option<usize>,
        row_ids: Vec<Uuid>,
        present: Vec<bool>,
        vectors: Vec<f32>,
    ) -> ILResult<()> {
        if row_ids.is_empty() {
            return Ok(());
        }
        match (self.dim, dim) {
            (Some(existing), Some(new)) if existing != new => {
                return Err(ILError::index(format!(
                    "HNSW vector dimension changed from {existing} to {new}"
                )));
            }
            (None, new) => self.dim = new,
            _ => {}
        }
        self.row_ids.extend(row_ids);
        self.present.extend(present);
        self.vectors.extend(vectors);
        Ok(())
    }

    fn take_graph(&mut self) -> ILResult<HnswGraph> {
        if let Some(graph) = self.graph.take() {
            return Ok(graph);
        }

        let dim = self.dim;
        let dim_value = dim.unwrap_or(0);
        let all_row_ids = std::mem::take(&mut self.row_ids);
        let present = std::mem::take(&mut self.present);
        let mut vectors = std::mem::take(&mut self.vectors);
        self.dim = None;

        let mut row_ids = Vec::with_capacity(all_row_ids.len());
        let mut row_offsets = Vec::with_capacity(all_row_ids.len());
        let mut write = 0usize;
        let mut read = 0usize;
        for (row_offset, (&is_present, &row_id)) in present.iter().zip(&all_row_ids).enumerate() {
            if !is_present {
                continue;
            }
            if row_offset >= u32::MAX as usize {
                return Err(ILError::index(
                    "HNSW segment exceeds the maximum number of rows",
                ));
            }
            row_ids.push(row_id);
            row_offsets.push(row_offset as u32);
            if write != read {
                vectors.copy_within(read..read + dim_value, write);
            }
            write += dim_value;
            read += dim_value;
        }
        vectors.truncate(write);

        HnswGraph::build(
            vectors,
            row_ids,
            row_offsets,
            dim_value,
            self.params.ef_construction,
        )
    }
}

fn row_id_at(row_id_array: &FixedSizeBinaryArray, index: usize) -> ILResult<Uuid> {
    if row_id_array.is_null(index) {
        return Err(ILError::index("HNSW row id must not be null"));
    }
    Uuid::from_slice(row_id_array.value(index)).map_err(Into::into)
}

#[async_trait::async_trait]
impl IndexBuilder for HnswIndexBuilder {
    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn is_empty(&self) -> bool {
        self.graph.is_none() && self.row_ids.is_empty()
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;
        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column: &ArrayRef = batch.column(key_column_index);

        if row_id_array.len() != key_column.len() {
            return Err(ILError::index(
                "HNSW row id array length does not match the key column",
            ));
        }

        match key_column.data_type() {
            DataType::List(inner) => {
                if !matches!(inner.data_type(), DataType::Float32) || inner.is_nullable() {
                    return Err(ILError::index(
                        "HNSW index key column must be a list of non-nullable float32",
                    ));
                }
                let list = key_column
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| ILError::index("HNSW key column is not a list array"))?;
                self.append_list_rows(&row_id_array, list)?;
            }
            DataType::FixedSizeList(inner, _) => {
                if !matches!(inner.data_type(), DataType::Float32) || inner.is_nullable() {
                    return Err(ILError::index(
                        "HNSW index key column must be a list of non-nullable float32",
                    ));
                }
                let list = key_column
                    .as_any()
                    .downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| {
                        ILError::index("HNSW key column is not a fixed size list array")
                    })?;
                self.append_fixed_size_list_rows(&row_id_array, list)?;
            }
            _ => {
                return Err(ILError::index(
                    "HNSW index key column must be a list of non-nullable float32",
                ));
            }
        }
        Ok(())
    }

    async fn read_file(&mut self, mut input_file: Box<dyn InputFile>) -> ILResult<()> {
        let file_meta = input_file.metadata().await?;
        let data = input_file.read(0..file_meta.size).await?;
        self.graph = Some(decode_graph(&data)?);
        Ok(())
    }

    async fn write_file(&mut self, mut output_file: Box<dyn OutputFile>) -> ILResult<()> {
        let graph = self.take_graph()?;
        let data = encode_graph(&graph)?;
        output_file.write(Bytes::from_owner(data)).await?;
        output_file.close().await?;
        Ok(())
    }

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()> {
        let pending = decode_pending(buf)?;
        self.merge_pending(
            pending.dim,
            pending.row_ids,
            pending.present,
            pending.vectors,
        )
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        write_pending(buf, self.dim, &self.row_ids, &self.present, &self.vectors)
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        let graph = self.take_graph()?;
        Ok(Box::new(HnswIndex::new(graph)))
    }
}
