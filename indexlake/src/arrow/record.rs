use std::sync::Arc;

use crate::arrow::{arrow_schema_to_schema, schema_to_arrow_schema};
use crate::record::{DataType, Scalar, SchemaRef};
use crate::{
    ILError, ILResult,
    record::{Row, Schema},
};
use arrow::array::{
    Array, BinaryArray, BinaryBuilder, BooleanArray, BooleanBuilder, Float32Array, Float32Builder,
    Float64Array, Float64Builder, Int16Array, Int16Builder, Int32Array, Int32Builder, Int64Array,
    Int64Builder, RecordBatch, StringArray, StringBuilder, make_builder,
};

macro_rules! builder_append {
    ($builder:expr, $builder_ty:ty, $field:expr, $row:expr, $row_method:ident, $index:expr, $convert:expr) => {{
        let builder = $builder
            .as_any_mut()
            .downcast_mut::<$builder_ty>()
            .unwrap_or_else(|| {
                panic!(
                    "Failed to downcast builder to {} for {:?}",
                    stringify!($builder_ty),
                    $field,
                )
            });
        let v = $row.$row_method($index).map_err(|e| {
            ILError::InternalError(format!(
                "Failed to get {} value for {:?}: {e:?}",
                stringify!($value_ty),
                $field,
            ))
        })?;

        match v {
            Some(v) => builder.append_value($convert(v)?),
            None => builder.append_null(),
        }
    }};
}

pub fn rows_to_arrow_record(schema: &Schema, rows: &[Row]) -> ILResult<RecordBatch> {
    let arrow_schema = schema_to_arrow_schema(schema)?;

    let mut array_builders = Vec::with_capacity(schema.fields.len());
    for arrow_field in arrow_schema.fields.iter() {
        array_builders.push(make_builder(arrow_field.data_type(), rows.len()));
    }

    for row in rows {
        for (i, field) in schema.fields.iter().enumerate() {
            match field.data_type {
                DataType::Int16 => {
                    builder_append!(array_builders[i], Int16Builder, field, row, int16, i, |v| {
                        Ok::<_, ILError>(v as i16)
                    });
                }
                DataType::Int32 => {
                    builder_append!(array_builders[i], Int32Builder, field, row, int32, i, |v| {
                        Ok::<_, ILError>(v as i32)
                    });
                }
                DataType::Int64 => {
                    builder_append!(array_builders[i], Int64Builder, field, row, int64, i, |v| {
                        Ok::<_, ILError>(v as i64)
                    });
                }
                DataType::Float32 => {
                    builder_append!(
                        array_builders[i],
                        Float32Builder,
                        field,
                        row,
                        float32,
                        i,
                        |v| Ok::<_, ILError>(v as f32)
                    );
                }
                DataType::Float64 => {
                    builder_append!(
                        array_builders[i],
                        Float64Builder,
                        field,
                        row,
                        float64,
                        i,
                        |v| Ok::<_, ILError>(v as f64)
                    );
                }
                DataType::Boolean => {
                    builder_append!(
                        array_builders[i],
                        BooleanBuilder,
                        field,
                        row,
                        boolean,
                        i,
                        |v| Ok::<_, ILError>(v as bool)
                    );
                }
                DataType::Utf8 => {
                    let convert: for<'a> fn(&'a String) -> ILResult<&'a String> =
                        |v: &String| Ok(v);
                    builder_append!(
                        array_builders[i],
                        StringBuilder,
                        field,
                        row,
                        utf8,
                        i,
                        convert
                    );
                }
                DataType::Binary => {
                    let convert: for<'a> fn(&'a Vec<u8>) -> ILResult<&'a Vec<u8>> =
                        |v: &Vec<u8>| Ok(v);
                    builder_append!(
                        array_builders[i],
                        BinaryBuilder,
                        field,
                        row,
                        binary,
                        i,
                        convert
                    );
                }
            }
        }
    }

    let columns = array_builders
        .into_iter()
        .map(|mut builder| builder.finish())
        .collect();
    RecordBatch::try_new(Arc::new(arrow_schema), columns)
        .map_err(|e| ILError::InternalError(format!("Failed to create record batch: {e:?}")))
}

pub fn record_batch_to_rows(record_batch: &RecordBatch, schema: SchemaRef) -> ILResult<Vec<Row>> {
    let mut column_arrays = Vec::with_capacity(schema.fields.len());
    let arrow_schema = record_batch.schema();
    for field in schema.fields.iter() {
        let arrow_col_idx = arrow_schema.index_of(&field.name).map_err(|_e| {
            ILError::InternalError(format!(
                "Failed to find field {field:?} in arrow schema: {arrow_schema:?}"
            ))
        })?;
        let any_array = record_batch.column(arrow_col_idx).as_any();
        let mut column_scalars = Vec::with_capacity(record_batch.num_rows());
        match field.data_type {
            DataType::Int16 => {
                let array = any_array.downcast_ref::<Int16Array>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to Int16Array"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Int16(v));
                }
            }
            DataType::Int32 => {
                let array = any_array.downcast_ref::<Int32Array>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to Int32Array"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Int32(v));
                }
            }
            DataType::Int64 => {
                let array = any_array.downcast_ref::<Int64Array>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to Int64Array"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Int64(v));
                }
            }
            DataType::Float32 => {
                let array = any_array.downcast_ref::<Float32Array>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to Float32Array"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Float32(v));
                }
            }
            DataType::Float64 => {
                let array = any_array.downcast_ref::<Float64Array>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to Float64Array"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Float64(v));
                }
            }
            DataType::Boolean => {
                let array = any_array.downcast_ref::<BooleanArray>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to BooleanArray"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Boolean(v));
                }
            }
            DataType::Utf8 => {
                let array = any_array.downcast_ref::<StringArray>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to StringArray"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Utf8(v.map(|v| v.to_string())));
                }
            }
            DataType::Binary => {
                let array = any_array.downcast_ref::<BinaryArray>().ok_or_else(|| {
                    ILError::InternalError(format!(
                        "Failed to downcast field {field:?} to BinaryArray"
                    ))
                })?;
                for v in array.iter() {
                    column_scalars.push(Scalar::Binary(v.map(|v| v.to_vec())));
                }
            }
        }
        column_arrays.push(column_scalars);
    }
    let mut rows = Vec::with_capacity(record_batch.num_rows());
    for row_idx in 0..record_batch.num_rows() {
        let mut values = Vec::with_capacity(schema.fields.len());
        for col_idx in 0..schema.fields.len() {
            values.push(std::mem::replace(
                &mut column_arrays[col_idx][row_idx],
                Scalar::Int32(None),
            ));
        }
        rows.push(Row::new(schema.clone(), values));
    }
    Ok(rows)
}
