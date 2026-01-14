use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeBinaryArray, RecordBatch, RecordBatchOptions,
};
use arrow::datatypes::{FieldRef, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow_schema::SchemaRef;
use uuid::Uuid;

use crate::catalog::{INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use crate::expr::{Expr, visited_columns};
use crate::{ILError, ILResult};

pub fn schema_without_row_id(schema: &Schema) -> Schema {
    let fields = schema
        .fields
        .iter()
        .filter(|field| field.name() != INTERNAL_ROW_ID_FIELD_NAME)
        .cloned()
        .collect::<Vec<_>>();
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

pub fn extract_row_id_array_from_record_batch(
    record_batch: &RecordBatch,
) -> ILResult<FixedSizeBinaryArray> {
    let index = record_batch
        .schema_ref()
        .index_of(INTERNAL_ROW_ID_FIELD_NAME)?;
    let row_id_array = record_batch
        .column(index)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| ILError::internal("Row id column is not an FixedSizeBinaryArray"))?;
    if row_id_array.value_length() != 16 {
        return Err(ILError::internal(
            "Row id column is not an FixedSizeBinaryArray with length 16",
        ));
    }
    Ok(row_id_array.clone())
}

pub fn extract_row_ids_from_record_batch(record_batch: &RecordBatch) -> ILResult<Vec<Uuid>> {
    let row_id_array = extract_row_id_array_from_record_batch(record_batch)?;
    fixed_size_binary_array_to_uuids(&row_id_array)
}

pub fn sort_record_batches(batches: &[RecordBatch], sort_col: &str) -> ILResult<Vec<RecordBatch>> {
    if batches.is_empty() {
        return Ok(Vec::new());
    }

    // TODO not concat batches, this might lead arrow array overflow
    let record = arrow::compute::concat_batches(&batches[0].schema(), batches)?;

    let sort_col_idx = record.schema().index_of(sort_col)?;
    let sort_array = record.column(sort_col_idx);

    let indices = arrow::compute::sort_to_indices(sort_array, None, None)?;

    let sorted_arrays: Vec<ArrayRef> = record
        .columns()
        .iter()
        .map(|col| arrow::compute::take(col, &indices, None))
        .collect::<arrow::error::Result<_>>()?;

    let options = RecordBatchOptions::new().with_row_count(Some(record.num_rows()));
    let batch = RecordBatch::try_new_with_options(record.schema(), sorted_arrays, &options)?;

    Ok(vec![batch])
}

pub fn array_to_uuids(array: &dyn Array) -> ILResult<Vec<Uuid>> {
    let data_type = array.data_type();
    let fixed_size_binary_array = array.as_fixed_size_binary_opt().ok_or_else(|| {
        ILError::internal(format!("array is not an FixedSizeBinaryArray: {data_type}",))
    })?;
    fixed_size_binary_array_to_uuids(fixed_size_binary_array)
}

pub fn fixed_size_binary_array_to_uuids(array: &FixedSizeBinaryArray) -> ILResult<Vec<Uuid>> {
    if array.value_length() != 16 {
        return Err(ILError::internal(
            "array is not an FixedSizeBinaryArray with length 16",
        ));
    }
    let mut uuids = Vec::with_capacity(array.len());
    for i in 0..array.len() {
        let uuid = Uuid::from_slice(array.value(i))?;
        uuids.push(uuid);
    }
    Ok(uuids)
}

pub fn build_row_id_array<T, U>(iter: T) -> ILResult<FixedSizeBinaryArray>
where
    T: Iterator<Item = U>,
    U: AsRef<[u8]>,
{
    let array =
        FixedSizeBinaryArray::try_from_sparse_iter_with_size(iter.map(|item| Some(item)), 16)?;
    Ok(array)
}

pub fn project_schema(schema: &Schema, projection: Option<&Vec<usize>>) -> ILResult<Schema> {
    if let Some(projection) = projection {
        let schema = schema.project(projection)?;
        Ok(schema)
    } else {
        Ok(schema.clone())
    }
}

pub fn build_projection_from_condition(schema: &Schema, condition: &Expr) -> ILResult<Vec<usize>> {
    let visited_columns = visited_columns(condition);
    if visited_columns.is_empty() {
        return Ok(Vec::new());
    }
    let mut projection = Vec::new();
    for col in visited_columns {
        let idx = schema.index_of(&col)?;
        projection.push(idx);
    }
    projection.sort();
    Ok(projection)
}

pub fn serialize_array(array: ArrayRef, field: FieldRef) -> ILResult<Vec<u8>> {
    let schema = Arc::new(Schema::new(vec![field]));
    let batch = RecordBatch::try_new(schema.clone(), vec![array.clone()])?;
    let mut buf = Vec::with_capacity(array.get_array_memory_size());
    let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(buf)
}

pub fn deserialize_array(buf: &[u8], field: FieldRef) -> ILResult<ArrayRef> {
    let schema = Arc::new(Schema::new(vec![field]));
    let reader = StreamReader::try_new(buf, None)?;
    if reader.schema() != schema {
        return Err(ILError::internal(format!(
            "Schema mismatch when deserializing array: {:?}, {:?}",
            reader.schema(),
            schema,
        )));
    }
    let mut arrays = Vec::new();
    for batch in reader {
        let batch = batch?;
        let array = batch.column(0).clone();
        arrays.push(array);
    }
    let array = arrow::compute::concat(
        arrays
            .iter()
            .map(|a| a.as_ref())
            .collect::<Vec<_>>()
            .as_slice(),
    )?;
    Ok(array)
}

pub fn rewrite_batch_schema(
    batch: &RecordBatch,
    field_name_id_map: &HashMap<String, Uuid>,
) -> ILResult<RecordBatch> {
    let mut new_fields = Vec::new();
    let batch_schema = batch.schema_ref();
    for field in batch_schema.fields() {
        if let Some(field_id) = field_name_id_map.get(field.name()) {
            let new_field_name = hex::encode(field_id);
            let new_field = field.as_ref().clone().with_name(new_field_name);
            new_fields.push(Arc::new(new_field));
        } else if field.name() == INTERNAL_ROW_ID_FIELD_NAME {
            new_fields.push(field.clone());
        } else {
            return Err(ILError::invalid_input(format!("Invalid field {field}")));
        }
    }
    let new_schema =
        Arc::new(Schema::new(new_fields).with_metadata(batch_schema.metadata().clone()));
    let new_batch = RecordBatch::try_new(new_schema, batch.columns().to_vec())?;
    Ok(new_batch)
}

pub fn correct_batch_schema(
    batch: &RecordBatch,
    field_id_name_map: &HashMap<Uuid, String>,
) -> ILResult<RecordBatch> {
    let mut new_fields = Vec::new();
    let batch_schema = batch.schema_ref();
    for field in batch_schema.fields() {
        if field.name() == INTERNAL_ROW_ID_FIELD_NAME {
            new_fields.push(field.clone());
        } else if let Ok(field_id) = Uuid::parse_str(field.name()) {
            let Some(correct_field_name) = field_id_name_map.get(&field_id) else {
                return Err(ILError::internal(format!(
                    "Not found field name for field id {field_id}"
                )));
            };
            let new_field = field.as_ref().clone().with_name(correct_field_name);
            new_fields.push(Arc::new(new_field));
        } else {
            new_fields.push(field.clone());
        }
    }
    let new_schema =
        Arc::new(Schema::new(new_fields).with_metadata(batch_schema.metadata().clone()));
    let new_batch = RecordBatch::try_new(new_schema, batch.columns().to_vec())?;
    Ok(new_batch)
}

pub(crate) fn complete_batch_missing_fields(
    batch: RecordBatch,
    target_schema: SchemaRef,
    field_id_default_value_map: &HashMap<Uuid, Scalar>,
) -> ILResult<RecordBatch> {
    let mut new_columns = Vec::with_capacity(target_schema.fields().len());

    let batch_schema = batch.schema_ref();

    for field in target_schema.fields() {
        let field_name = field.name();

        if let Ok(index) = batch_schema.index_of(field_name) {
            new_columns.push(batch.column(index).clone());
        } else {
            let Ok(field_id) = Uuid::parse_str(field.name()) else {
                return Err(ILError::internal(format!(
                    "Failed to parse field name {field_name} as uuid",
                )));
            };
            if let Some(default_value) = field_id_default_value_map.get(&field_id) {
                let array = default_value.to_array_of_size(batch.num_rows())?;
                new_columns.push(array);
            } else {
                return Err(ILError::internal(
                    "No default value for field {field_name} to complete record batch",
                ));
            }
        }
    }
    let new_batch = RecordBatch::try_new(target_schema, new_columns)?;
    Ok(new_batch)
}

pub fn timestamp_ms_from_now(duration: Duration) -> i64 {
    let start_time = SystemTime::now();
    let end_time = start_time + duration;
    end_time
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as i64
}
