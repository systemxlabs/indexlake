use std::iter::repeat_n;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeBinaryArray, RecordBatch, RecordBatchOptions, StringArray,
};
use arrow::datatypes::{FieldRef, Schema};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use uuid::Uuid;

use crate::catalog::INTERNAL_ROW_ID_FIELD_NAME;
use crate::expr::{Expr, visited_columns};
use crate::table::MetadataColumn;
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

pub fn append_new_fields_to_schema(schema: &Schema, new_fields: &[FieldRef]) -> Schema {
    let mut fields = schema.fields.to_vec();
    fields.extend(new_fields.iter().cloned());
    Schema::new_with_metadata(fields, schema.metadata().clone())
}

pub fn record_batch_with_location_kind(
    record_batch: &RecordBatch,
    location_kind: &str,
) -> ILResult<RecordBatch> {
    let new_schema = append_new_fields_to_schema(
        record_batch.schema_ref(),
        &[MetadataColumn::LocationKind.to_field()],
    );

    let mut arrays = record_batch.columns().to_vec();
    let location_kind_array = Arc::new(StringArray::from_iter_values(repeat_n(
        location_kind,
        record_batch.num_rows(),
    )));
    arrays.push(location_kind_array);

    let options = RecordBatchOptions::new().with_row_count(Some(record_batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(new_schema),
        arrays,
        &options,
    )?)
}

pub fn record_batch_with_location(
    record_batch: &RecordBatch,
    location: &str,
) -> ILResult<RecordBatch> {
    let new_schema = append_new_fields_to_schema(
        record_batch.schema_ref(),
        &[MetadataColumn::Location.to_field()],
    );

    let mut arrays = record_batch.columns().to_vec();
    let location_array = Arc::new(StringArray::from_iter_values(repeat_n(
        location,
        record_batch.num_rows(),
    )));
    arrays.push(location_array);

    let options = RecordBatchOptions::new().with_row_count(Some(record_batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        Arc::new(new_schema),
        arrays,
        &options,
    )?)
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
