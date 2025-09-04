use std::fmt::Display;

use crate::catalog::{CatalogSchemaRef, INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use crate::utils::deserialize_array;
use crate::{ILError, ILResult};
use arrow::array::*;
use arrow::datatypes::{DataType, SchemaRef, TimeUnit, i256};
use uuid::Uuid;

#[derive(Debug)]
pub struct Row {
    pub schema: CatalogSchemaRef,
    pub values: Vec<Scalar>,
}

impl Row {
    pub fn new(schema: CatalogSchemaRef, values: Vec<Scalar>) -> Self {
        assert_eq!(schema.columns.len(), values.len());
        Self { schema, values }
    }

    pub fn get_row_id(&self) -> ILResult<Option<Uuid>> {
        let Some(idx) = self.schema.index_of(INTERNAL_ROW_ID_FIELD_NAME) else {
            return Ok(None);
        };
        self.uuid(idx)
    }

    pub fn int8(&self, index: usize) -> ILResult<Option<i8>> {
        match self.values[index] {
            Scalar::Int8(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Int8 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn int16(&self, index: usize) -> ILResult<Option<i16>> {
        match self.values[index] {
            Scalar::Int16(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Int16 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn int32(&self, index: usize) -> ILResult<Option<i32>> {
        match self.values[index] {
            Scalar::Int32(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Int32 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn int64(&self, index: usize) -> ILResult<Option<i64>> {
        match self.values[index] {
            Scalar::Int64(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected BigInt at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn uint8(&self, index: usize) -> ILResult<Option<u8>> {
        match self.values[index] {
            Scalar::UInt8(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected UInt8 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn uint16(&self, index: usize) -> ILResult<Option<u16>> {
        match self.values[index] {
            Scalar::UInt16(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected UInt16 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn uint32(&self, index: usize) -> ILResult<Option<u32>> {
        match self.values[index] {
            Scalar::UInt32(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected UInt32 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn uint64(&self, index: usize) -> ILResult<Option<u64>> {
        match self.values[index] {
            Scalar::UInt64(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected UInt64 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn float32(&self, index: usize) -> ILResult<Option<f32>> {
        match self.values[index] {
            Scalar::Float32(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Float32 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn float64(&self, index: usize) -> ILResult<Option<f64>> {
        match self.values[index] {
            Scalar::Float64(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Float64 at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn utf8(&self, index: usize) -> ILResult<Option<&String>> {
        match &self.values[index] {
            Scalar::Utf8(v) => Ok(v.as_ref()),
            _ => Err(ILError::internal(format!(
                "Expected Varchar at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn utf8_owned(&mut self, index: usize) -> ILResult<Option<String>> {
        match &mut self.values[index] {
            Scalar::Utf8(v) => Ok(std::mem::take(v)),
            _ => Err(ILError::internal(format!(
                "Expected Varchar at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn binary(&self, index: usize) -> ILResult<Option<&Vec<u8>>> {
        match &self.values[index] {
            Scalar::Binary(v) => Ok(v.as_ref()),
            _ => Err(ILError::internal(format!(
                "Expected Binary at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn binary_owned(&mut self, index: usize) -> ILResult<Option<Vec<u8>>> {
        match &mut self.values[index] {
            Scalar::Binary(v) => Ok(std::mem::take(v)),
            _ => Err(ILError::internal(format!(
                "Expected Binary at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn uuid(&self, index: usize) -> ILResult<Option<Uuid>> {
        match &self.values[index] {
            Scalar::Binary(Some(v)) => {
                let uuid = Uuid::from_slice(v)?;
                Ok(Some(uuid))
            }
            Scalar::Binary(None) => Ok(None),
            _ => Err(ILError::internal(format!(
                "Expected Binary at index {index} for row {self:?}"
            ))),
        }
    }

    pub fn boolean(&self, index: usize) -> ILResult<Option<bool>> {
        match self.values[index] {
            Scalar::Boolean(v) => Ok(v),
            _ => Err(ILError::internal(format!(
                "Expected Boolean at index {index} for row {self:?}"
            ))),
        }
    }
}

pub fn pretty_print_rows(schema_opt: Option<CatalogSchemaRef>, rows: &[Row]) -> impl Display {
    let mut table = comfy_table::Table::new();
    table.load_preset("||--+-++|    ++++++");

    let schema_opt = schema_opt.or_else(|| {
        if rows.is_empty() {
            None
        } else {
            Some(rows[0].schema.clone())
        }
    });
    if let Some(schema) = schema_opt {
        let mut header = Vec::new();
        for field in schema.columns.iter() {
            header.push(field.name.clone());
        }
        table.set_header(header);
    }

    if rows.is_empty() {
        return table;
    }

    for row in rows {
        let mut cells = Vec::new();
        for value in row.values.iter() {
            cells.push(value.to_string());
        }
        table.add_row(cells);
    }
    table
}

macro_rules! builder_append {
    (
        $builder:expr,
        $builder_ty:ty,
        $field:expr,
        $row:expr,
        $row_method:ident,
        $index:expr,
        $convert:expr
    ) => {{
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
            ILError::internal(format!(
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

macro_rules! list_builder_append {
    ($builder:expr, $builder_ty:ty, $field:expr, $inner_field:expr, $row:expr, $index:expr) => {{
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
        let v = $row.binary($index).map_err(|e| {
            ILError::internal(format!(
                "Failed to get {} value for {:?}: {e:?}",
                stringify!($value_ty),
                $field,
            ))
        })?;

        match v {
            Some(v) => {
                let values_builder = builder.values();
                let array = deserialize_array(&v, $inner_field.clone())?;
                match $inner_field.data_type() {
                    DataType::Boolean => {
                        values_builder_append!(values_builder, BooleanBuilder, array, BooleanArray);
                    }
                    DataType::Int8 => {
                        values_builder_append!(values_builder, Int8Builder, array, Int8Array);
                    }
                    DataType::Int16 => {
                        values_builder_append!(values_builder, Int16Builder, array, Int16Array);
                    }
                    DataType::Int32 => {
                        values_builder_append!(values_builder, Int32Builder, array, Int32Array);
                    }
                    DataType::Int64 => {
                        values_builder_append!(values_builder, Int64Builder, array, Int64Array);
                    }
                    DataType::UInt8 => {
                        values_builder_append!(values_builder, UInt8Builder, array, UInt8Array);
                    }
                    DataType::UInt16 => {
                        values_builder_append!(values_builder, UInt16Builder, array, UInt16Array);
                    }
                    DataType::UInt32 => {
                        values_builder_append!(values_builder, UInt32Builder, array, UInt32Array);
                    }
                    DataType::UInt64 => {
                        values_builder_append!(values_builder, UInt64Builder, array, UInt64Array);
                    }
                    DataType::Float32 => {
                        values_builder_append!(values_builder, Float32Builder, array, Float32Array);
                    }
                    DataType::Float64 => {
                        values_builder_append!(values_builder, Float64Builder, array, Float64Array);
                    }
                    DataType::Timestamp(TimeUnit::Second, _) => {
                        values_builder_append!(
                            values_builder,
                            TimestampSecondBuilder,
                            array,
                            TimestampSecondArray
                        );
                    }
                    DataType::Timestamp(TimeUnit::Millisecond, _) => {
                        values_builder_append!(
                            values_builder,
                            TimestampMillisecondBuilder,
                            array,
                            TimestampMillisecondArray
                        );
                    }
                    DataType::Timestamp(TimeUnit::Microsecond, _) => {
                        values_builder_append!(
                            values_builder,
                            TimestampMicrosecondBuilder,
                            array,
                            TimestampMicrosecondArray
                        );
                    }
                    DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                        values_builder_append!(
                            values_builder,
                            TimestampNanosecondBuilder,
                            array,
                            TimestampNanosecondArray
                        );
                    }
                    DataType::Date32 => {
                        values_builder_append!(values_builder, Date32Builder, array, Date32Array);
                    }
                    DataType::Date64 => {
                        values_builder_append!(values_builder, Date64Builder, array, Date64Array);
                    }
                    DataType::Time32(TimeUnit::Second) => {
                        values_builder_append!(
                            values_builder,
                            Time32SecondBuilder,
                            array,
                            Time32SecondArray
                        );
                    }
                    DataType::Time32(TimeUnit::Millisecond) => {
                        values_builder_append!(
                            values_builder,
                            Time32MillisecondBuilder,
                            array,
                            Time32MillisecondArray
                        );
                    }
                    DataType::Time64(TimeUnit::Microsecond) => {
                        values_builder_append!(
                            values_builder,
                            Time64MicrosecondBuilder,
                            array,
                            Time64MicrosecondArray
                        );
                    }
                    DataType::Time64(TimeUnit::Nanosecond) => {
                        values_builder_append!(
                            values_builder,
                            Time64NanosecondBuilder,
                            array,
                            Time64NanosecondArray
                        );
                    }
                    DataType::Binary => {
                        values_builder_append!(values_builder, BinaryBuilder, array, BinaryArray);
                    }
                    // TODO FixedSizeBinaryBuilder supports append_array
                    DataType::LargeBinary => {
                        values_builder_append!(
                            values_builder,
                            LargeBinaryBuilder,
                            array,
                            LargeBinaryArray
                        );
                    }
                    DataType::Utf8 => {
                        values_builder_append!(values_builder, StringBuilder, array, StringArray);
                    }
                    DataType::LargeUtf8 => {
                        values_builder_append!(
                            values_builder,
                            LargeStringBuilder,
                            array,
                            LargeStringArray
                        );
                    }
                    DataType::Decimal128(..) => {
                        values_builder_append!(
                            values_builder,
                            Decimal128Builder,
                            array,
                            Decimal128Array
                        );
                    }
                    DataType::Decimal256(..) => {
                        values_builder_append!(
                            values_builder,
                            Decimal256Builder,
                            array,
                            Decimal256Array
                        );
                    }
                    _ => {
                        return Err(ILError::not_supported(format!(
                            "Not supported data type: {}",
                            $field.data_type()
                        )));
                    }
                }
                builder.append(true);
            }
            None => builder.append(false),
        }
    }};
}

macro_rules! values_builder_append {
    ($builder:expr, $builder_ty:ty, $array:expr, $array_ty:ty) => {{
        let values_builder = $builder
            .as_any_mut()
            .downcast_mut::<$builder_ty>()
            .unwrap_or_else(|| {
                panic!(
                    "Failed to downcast values builder to {}",
                    stringify!($builder_ty),
                )
            });

        let array = $array.as_any().downcast_ref::<$array_ty>().ok_or_else(|| {
            ILError::internal(format!(
                "Failed to downcast inner array to {}",
                stringify!($array_ty),
            ))
        })?;
        values_builder.append_array(array);
    }};
}

macro_rules! fixed_size_list_values_builder_append {
    (
        $data:expr,
        $builder:expr,
        $values_builder_ty:ty,
        $inner_field:expr,
        $array_ty:ty,
        $len:expr
    ) => {{
        match $data {
            Some(v) => {
                let array = deserialize_array(&v, $inner_field.clone())?;
                let values_builder = $builder.values();
                values_builder_append!(values_builder, $values_builder_ty, array, $array_ty);
                $builder.append(true);
            }
            None => {
                let values_builder = $builder.values();
                values_builder_append_nulls!(values_builder, $values_builder_ty, $len);
                $builder.append(false);
            }
        }
    }};
}

macro_rules! values_builder_append_nulls {
    ($builder:expr, $builder_ty:ty, $count:expr) => {{
        let values_builder = $builder
            .as_any_mut()
            .downcast_mut::<$builder_ty>()
            .unwrap_or_else(|| {
                panic!(
                    "Failed to downcast values builder to {}",
                    stringify!($builder_ty),
                )
            });

        for _ in 0..$count {
            values_builder.append_null();
        }
    }};
}

pub fn rows_to_record_batch(schema: &SchemaRef, rows: &[Row]) -> ILResult<RecordBatch> {
    let mut array_builders = Vec::with_capacity(schema.fields.len());
    for field in schema.fields.iter() {
        // TODO wait upstream to fix this
        let builder = match field.data_type() {
            DataType::BinaryView => {
                Box::new(BinaryViewBuilder::with_capacity(rows.len())) as Box<dyn ArrayBuilder>
            }
            DataType::Utf8View => {
                Box::new(StringViewBuilder::with_capacity(rows.len())) as Box<dyn ArrayBuilder>
            }
            _ => make_builder(field.data_type(), rows.len()),
        };
        array_builders.push(builder);
    }

    let binary_convert: for<'a> fn(&'a Vec<u8>) -> ILResult<&'a Vec<u8>> = |v: &Vec<u8>| Ok(v);
    let string_convert: for<'a> fn(&'a String) -> ILResult<&'a String> = |v: &String| Ok(v);
    for row in rows {
        for (i, field) in schema.fields.iter().enumerate() {
            match field.data_type() {
                DataType::Boolean => {
                    builder_append!(
                        array_builders[i],
                        BooleanBuilder,
                        field,
                        row,
                        boolean,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Int8 => {
                    builder_append!(
                        array_builders[i],
                        Int8Builder,
                        field,
                        row,
                        int8,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Int16 => {
                    builder_append!(
                        array_builders[i],
                        Int16Builder,
                        field,
                        row,
                        int16,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Int32 => {
                    builder_append!(
                        array_builders[i],
                        Int32Builder,
                        field,
                        row,
                        int32,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Int64 => {
                    builder_append!(
                        array_builders[i],
                        Int64Builder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::UInt8 => {
                    builder_append!(
                        array_builders[i],
                        UInt8Builder,
                        field,
                        row,
                        uint8,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::UInt16 => {
                    builder_append!(
                        array_builders[i],
                        UInt16Builder,
                        field,
                        row,
                        uint16,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::UInt32 => {
                    builder_append!(
                        array_builders[i],
                        UInt32Builder,
                        field,
                        row,
                        uint32,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::UInt64 => {
                    builder_append!(
                        array_builders[i],
                        UInt64Builder,
                        field,
                        row,
                        uint64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Float32 => {
                    builder_append!(
                        array_builders[i],
                        Float32Builder,
                        field,
                        row,
                        float32,
                        i,
                        Ok::<_, ILError>
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
                        Ok::<_, ILError>
                    );
                }
                DataType::Timestamp(TimeUnit::Second, _) => {
                    builder_append!(
                        array_builders[i],
                        TimestampSecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Timestamp(TimeUnit::Millisecond, _) => {
                    builder_append!(
                        array_builders[i],
                        TimestampMillisecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Timestamp(TimeUnit::Microsecond, _) => {
                    builder_append!(
                        array_builders[i],
                        TimestampMicrosecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        |v| { Ok::<_, ILError>(v) }
                    );
                }
                DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                    builder_append!(
                        array_builders[i],
                        TimestampNanosecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Date32 => {
                    builder_append!(
                        array_builders[i],
                        Date32Builder,
                        field,
                        row,
                        int32,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Date64 => {
                    builder_append!(
                        array_builders[i],
                        Date64Builder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Time32(TimeUnit::Second) => {
                    builder_append!(
                        array_builders[i],
                        Time32SecondBuilder,
                        field,
                        row,
                        int32,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Time32(TimeUnit::Millisecond) => {
                    builder_append!(
                        array_builders[i],
                        Time32MillisecondBuilder,
                        field,
                        row,
                        int32,
                        i,
                        |v| { Ok::<_, ILError>(v) }
                    );
                }
                DataType::Time64(TimeUnit::Microsecond) => {
                    builder_append!(
                        array_builders[i],
                        Time64MicrosecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Time64(TimeUnit::Nanosecond) => {
                    builder_append!(
                        array_builders[i],
                        Time64NanosecondBuilder,
                        field,
                        row,
                        int64,
                        i,
                        Ok::<_, ILError>
                    );
                }
                DataType::Binary => {
                    builder_append!(
                        array_builders[i],
                        BinaryBuilder,
                        field,
                        row,
                        binary,
                        i,
                        binary_convert
                    );
                }
                DataType::FixedSizeBinary(_) => {
                    let builder = array_builders[i]
                        .as_any_mut()
                        .downcast_mut::<FixedSizeBinaryBuilder>()
                        .unwrap_or_else(|| {
                            panic!(
                                "Failed to downcast builder to FixedSizeBinaryBuilder for {field:?}"
                            )
                        });
                    let v = row.binary(i).map_err(|e| {
                        ILError::internal(format!(
                            "Failed to get binary value for {field:?}: {e:?}"
                        ))
                    })?;

                    match v {
                        Some(v) => builder.append_value(v)?,
                        None => builder.append_null(),
                    }
                }
                DataType::LargeBinary => {
                    builder_append!(
                        array_builders[i],
                        LargeBinaryBuilder,
                        field,
                        row,
                        binary,
                        i,
                        binary_convert
                    );
                }
                DataType::BinaryView => {
                    builder_append!(
                        array_builders[i],
                        BinaryViewBuilder,
                        field,
                        row,
                        binary,
                        i,
                        binary_convert
                    );
                }
                DataType::Utf8 => {
                    builder_append!(
                        array_builders[i],
                        StringBuilder,
                        field,
                        row,
                        utf8,
                        i,
                        string_convert
                    );
                }
                DataType::LargeUtf8 => {
                    builder_append!(
                        array_builders[i],
                        LargeStringBuilder,
                        field,
                        row,
                        utf8,
                        i,
                        string_convert
                    );
                }
                DataType::Utf8View => {
                    builder_append!(
                        array_builders[i],
                        StringViewBuilder,
                        field,
                        row,
                        utf8,
                        i,
                        string_convert
                    );
                }
                DataType::List(inner_field) => {
                    list_builder_append!(
                        array_builders[i],
                        ListBuilder<Box<dyn ArrayBuilder>>,
                        field,
                        inner_field,
                        row,
                        i
                    );
                }
                DataType::FixedSizeList(inner_field, len) => {
                    let builder = array_builders[i]
                        .as_any_mut()
                        .downcast_mut::<FixedSizeListBuilder<Box<dyn ArrayBuilder>>>()
                        .unwrap_or_else(|| {
                            panic!(
                                "Failed to downcast builder to FixedSizeListBuilder for {field:?}"
                            )
                        });
                    let v = row.binary(i).map_err(|e| {
                        ILError::internal(format!(
                            "Failed to get binary value for {field:?}: {e:?}"
                        ))
                    })?;

                    match inner_field.data_type() {
                        DataType::Boolean => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                BooleanBuilder,
                                inner_field,
                                BooleanArray,
                                *len
                            );
                        }
                        DataType::Int8 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                Int8Builder,
                                inner_field,
                                Int8Array,
                                *len
                            );
                        }
                        DataType::Int16 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                Int16Builder,
                                inner_field,
                                Int16Array,
                                *len
                            );
                        }
                        DataType::Int32 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                Int32Builder,
                                inner_field,
                                Int32Array,
                                *len
                            );
                        }
                        DataType::Int64 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                Int64Builder,
                                inner_field,
                                Int64Array,
                                *len
                            );
                        }
                        DataType::UInt8 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                UInt8Builder,
                                inner_field,
                                UInt8Array,
                                *len
                            );
                        }
                        DataType::UInt16 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                UInt16Builder,
                                inner_field,
                                UInt16Array,
                                *len
                            );
                        }
                        DataType::UInt32 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                UInt32Builder,
                                inner_field,
                                UInt32Array,
                                *len
                            );
                        }
                        DataType::UInt64 => {
                            fixed_size_list_values_builder_append!(
                                v,
                                builder,
                                UInt64Builder,
                                inner_field,
                                UInt64Array,
                                *len
                            );
                        }
                        _ => {
                            return Err(ILError::not_supported(format!(
                                "Not supported data type: {}",
                                inner_field.data_type()
                            )));
                        }
                    }
                }
                DataType::LargeList(inner_field) => {
                    list_builder_append!(
                        array_builders[i],
                        LargeListBuilder<Box<dyn ArrayBuilder>>,
                        field,
                        inner_field,
                        row,
                        i
                    );
                }
                DataType::Decimal128(..) => {
                    builder_append!(
                        array_builders[i],
                        Decimal128Builder,
                        field,
                        row,
                        utf8,
                        i,
                        |v: &String| {
                            let v = v.parse::<i128>().map_err(|e| {
                                ILError::internal(format!(
                                    "Failed to parse decimal128 value: {e:?}"
                                ))
                            })?;
                            Ok::<_, ILError>(v)
                        }
                    );
                }
                DataType::Decimal256(..) => {
                    builder_append!(
                        array_builders[i],
                        Decimal256Builder,
                        field,
                        row,
                        utf8,
                        i,
                        |v: &String| {
                            let v = v.parse::<i256>().map_err(|e| {
                                ILError::internal(format!(
                                    "Failed to parse decimal256 value: {e:?}"
                                ))
                            })?;
                            Ok::<_, ILError>(v)
                        }
                    );
                }
                _ => {
                    return Err(ILError::not_supported(format!(
                        "Not supported data type: {}",
                        field.data_type()
                    )));
                }
            }
        }
    }

    let columns = array_builders
        .into_iter()
        .map(|mut builder| builder.finish())
        .collect();
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    Ok(batch)
}
