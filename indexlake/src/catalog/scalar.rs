use std::cmp::Ordering;
use std::fmt::Display;
use std::iter::repeat_n;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, BinaryArray, BinaryViewArray, BooleanArray, Date32Array, Date64Array,
    Decimal128Array, Decimal256Array, FixedSizeBinaryArray, FixedSizeListArray, Float32Array,
    Float64Array, GenericListArray, Int8Array, Int16Array, Int32Array, Int64Array,
    LargeBinaryArray, LargeListArray, LargeStringArray, ListArray, RecordBatch, StringArray,
    StringViewArray, Time32MillisecondArray, Time32SecondArray, Time64MicrosecondArray,
    Time64NanosecondArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array, new_null_array,
};
use arrow::buffer::OffsetBuffer;
use arrow::compute::CastOptions;
use arrow::datatypes::{
    DataType, Date32Type, Date64Type, Decimal128Type, Decimal256Type, DecimalType, Float32Type,
    Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, Time32MillisecondType,
    Time32SecondType, Time64MicrosecondType, Time64NanosecondType, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type, i256,
};
use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_schema::{Field, Schema, TimeUnit};
use uuid::Uuid;

use crate::{ILError, ILResult, catalog::CatalogDatabase};

#[derive(Debug, Clone)]
pub enum Scalar {
    Boolean(Option<bool>),
    Int8(Option<i8>),
    Int16(Option<i16>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    UInt8(Option<u8>),
    UInt16(Option<u16>),
    UInt32(Option<u32>),
    UInt64(Option<u64>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    Utf8View(Option<String>),
    LargeUtf8(Option<String>),
    Binary(Option<Vec<u8>>),
    BinaryView(Option<Vec<u8>>),
    FixedSizeBinary(i32, Option<Vec<u8>>),
    LargeBinary(Option<Vec<u8>>),
    TimestampSecond(Option<i64>, Option<Arc<str>>),
    TimestampMillisecond(Option<i64>, Option<Arc<str>>),
    TimestampMicrosecond(Option<i64>, Option<Arc<str>>),
    TimestampNanosecond(Option<i64>, Option<Arc<str>>),
    Date32(Option<i32>),
    Date64(Option<i64>),
    Time32Second(Option<i32>),
    Time32Millisecond(Option<i32>),
    Time64Microsecond(Option<i64>),
    Time64Nanosecond(Option<i64>),
    List(Arc<ListArray>),
    FixedSizeList(Arc<FixedSizeListArray>),
    LargeList(Arc<LargeListArray>),
    Decimal128(Option<i128>, u8, i8),
    Decimal256(Option<i256>, u8, i8),
}

impl Scalar {
    pub fn try_new_null(data_type: &DataType) -> ILResult<Self> {
        Ok(match data_type {
            DataType::Boolean => Scalar::Boolean(None),
            DataType::Int8 => Scalar::Int8(None),
            DataType::Int16 => Scalar::Int16(None),
            DataType::Int32 => Scalar::Int32(None),
            DataType::Int64 => Scalar::Int64(None),
            DataType::UInt8 => Scalar::UInt8(None),
            DataType::UInt16 => Scalar::UInt16(None),
            DataType::UInt32 => Scalar::UInt32(None),
            DataType::UInt64 => Scalar::UInt64(None),
            DataType::Float32 => Scalar::Float32(None),
            DataType::Float64 => Scalar::Float64(None),
            DataType::Utf8 => Scalar::Utf8(None),
            DataType::Utf8View => Scalar::Utf8View(None),
            DataType::LargeUtf8 => Scalar::LargeUtf8(None),
            DataType::Binary => Scalar::Binary(None),
            DataType::BinaryView => Scalar::BinaryView(None),
            DataType::FixedSizeBinary(size) => Scalar::FixedSizeBinary(*size, None),
            DataType::LargeBinary => Scalar::LargeBinary(None),
            DataType::Timestamp(TimeUnit::Second, tz) => Scalar::TimestampSecond(None, tz.clone()),
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                Scalar::TimestampMillisecond(None, tz.clone())
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                Scalar::TimestampMicrosecond(None, tz.clone())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                Scalar::TimestampNanosecond(None, tz.clone())
            }
            DataType::Date32 => Scalar::Date32(None),
            DataType::Date64 => Scalar::Date64(None),
            DataType::Time32(TimeUnit::Second) => Scalar::Time32Second(None),
            DataType::Time32(TimeUnit::Millisecond) => Scalar::Time32Millisecond(None),
            DataType::Time64(TimeUnit::Microsecond) => Scalar::Time64Microsecond(None),
            DataType::Time64(TimeUnit::Nanosecond) => Scalar::Time64Nanosecond(None),
            DataType::List(field_ref) => {
                Scalar::List(Arc::new(GenericListArray::new_null(field_ref.clone(), 1)))
            }
            DataType::FixedSizeList(field_ref, size) => Scalar::FixedSizeList(Arc::new(
                FixedSizeListArray::new_null(field_ref.clone(), *size, 1),
            )),
            DataType::LargeList(field_ref) => {
                Scalar::LargeList(Arc::new(LargeListArray::new_null(field_ref.clone(), 1)))
            }
            DataType::Decimal128(precision, scale) => Scalar::Decimal128(None, *precision, *scale),
            DataType::Decimal256(precision, scale) => Scalar::Decimal256(None, *precision, *scale),
            _ => {
                return Err(ILError::not_supported(format!(
                    "Cannot create null scalar for data type: {data_type}",
                )));
            }
        })
    }
    pub fn is_null(&self) -> bool {
        match self {
            Scalar::Boolean(v) => v.is_none(),
            Scalar::Int8(v) => v.is_none(),
            Scalar::Int16(v) => v.is_none(),
            Scalar::Int32(v) => v.is_none(),
            Scalar::Int64(v) => v.is_none(),
            Scalar::UInt8(v) => v.is_none(),
            Scalar::UInt16(v) => v.is_none(),
            Scalar::UInt32(v) => v.is_none(),
            Scalar::UInt64(v) => v.is_none(),
            Scalar::Float32(v) => v.is_none(),
            Scalar::Float64(v) => v.is_none(),
            Scalar::Utf8(v) | Scalar::Utf8View(v) | Scalar::LargeUtf8(v) => v.is_none(),
            Scalar::Binary(v)
            | Scalar::BinaryView(v)
            | Scalar::FixedSizeBinary(_, v)
            | Scalar::LargeBinary(v) => v.is_none(),
            Scalar::TimestampSecond(v, _)
            | Scalar::TimestampMillisecond(v, _)
            | Scalar::TimestampMicrosecond(v, _)
            | Scalar::TimestampNanosecond(v, _) => v.is_none(),
            Scalar::Date32(v) => v.is_none(),
            Scalar::Date64(v) => v.is_none(),
            Scalar::Time32Second(v) | Scalar::Time32Millisecond(v) => v.is_none(),
            Scalar::Time64Microsecond(v) | Scalar::Time64Nanosecond(v) => v.is_none(),
            Scalar::List(v) => v.len() == v.null_count(),
            Scalar::FixedSizeList(v) => v.len() == v.null_count(),
            Scalar::LargeList(v) => v.len() == v.null_count(),
            Scalar::Decimal128(v, _, _) => v.is_none(),
            Scalar::Decimal256(v, _, _) => v.is_none(),
        }
    }

    pub fn as_bool(&self) -> ILResult<Option<bool>> {
        match self {
            Scalar::Boolean(v) => Ok(*v),
            _ => Err(ILError::invalid_input(format!(
                "Expected boolean, got {self:?}",
            ))),
        }
    }

    pub fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        if self.is_null() {
            return Ok("null".to_string());
        }
        match self {
            Scalar::Boolean(_)
            | Scalar::Int8(_)
            | Scalar::Int16(_)
            | Scalar::Int32(_)
            | Scalar::Int64(_)
            | Scalar::UInt8(_)
            | Scalar::UInt16(_)
            | Scalar::UInt32(_)
            | Scalar::UInt64(_)
            | Scalar::Float32(_)
            | Scalar::Float64(_) => Ok(self.to_string()),
            Scalar::Utf8(v) | Scalar::Utf8View(v) | Scalar::LargeUtf8(v) => match v {
                Some(value) => Ok(database.sql_string_literal(value)),
                None => Ok("null".to_string()),
            },
            Scalar::Binary(v) | Scalar::BinaryView(v) | Scalar::LargeBinary(v) => match v {
                Some(value) => Ok(database.sql_binary_literal(value)),
                None => Ok("null".to_string()),
            },
            Scalar::FixedSizeBinary(size, v) => match v {
                Some(value) => {
                    if *size == 16 {
                        Ok(database.sql_uuid_literal(&Uuid::from_slice(value)?))
                    } else {
                        Ok(database.sql_binary_literal(value))
                    }
                }
                None => Ok("null".to_string()),
            },
            Scalar::Decimal128(v, precision, scale) => match v {
                Some(value) => Ok(Decimal128Type::format_decimal(*value, *precision, *scale)),
                None => Ok("null".to_string()),
            },
            Scalar::Decimal256(v, precision, scale) => match v {
                Some(value) => Ok(Decimal256Type::format_decimal(*value, *precision, *scale)),
                None => Ok("null".to_string()),
            },
            Scalar::TimestampSecond(_, _)
            | Scalar::TimestampMillisecond(_, _)
            | Scalar::TimestampMicrosecond(_, _)
            | Scalar::TimestampNanosecond(_, _)
            | Scalar::Date32(_)
            | Scalar::Date64(_)
            | Scalar::Time32Second(_)
            | Scalar::Time32Millisecond(_)
            | Scalar::Time64Microsecond(_)
            | Scalar::Time64Nanosecond(_)
            | Scalar::List(_)
            | Scalar::FixedSizeList(_)
            | Scalar::LargeList(_) => Err(ILError::not_supported(
                "Not supported to convert scalar {self:?} to sql",
            )),
        }
    }

    pub fn to_arrow_scalar(&self) -> ILResult<arrow::array::Scalar<ArrayRef>> {
        Ok(arrow::array::Scalar::new(self.to_array_of_size(1)?))
    }

    pub fn to_array_of_size(&self, size: usize) -> ILResult<ArrayRef> {
        Ok(match self {
            Scalar::Boolean(e) => Arc::new(BooleanArray::from(vec![*e; size])) as ArrayRef,
            Scalar::Int8(e) => match e {
                Some(value) => Arc::new(Int8Array::from_value(*value, size)),
                None => new_null_array(&DataType::Int8, size),
            },
            Scalar::Int16(e) => match e {
                Some(value) => Arc::new(Int16Array::from_value(*value, size)),
                None => new_null_array(&DataType::Int16, size),
            },
            Scalar::Int32(e) => match e {
                Some(value) => Arc::new(Int32Array::from_value(*value, size)),
                None => new_null_array(&DataType::Int32, size),
            },
            Scalar::Int64(e) => match e {
                Some(value) => Arc::new(Int64Array::from_value(*value, size)),
                None => new_null_array(&DataType::Int64, size),
            },
            Scalar::UInt8(e) => match e {
                Some(value) => Arc::new(UInt8Array::from_value(*value, size)),
                None => new_null_array(&DataType::UInt8, size),
            },
            Scalar::UInt16(e) => match e {
                Some(value) => Arc::new(UInt16Array::from_value(*value, size)),
                None => new_null_array(&DataType::UInt16, size),
            },
            Scalar::UInt32(e) => match e {
                Some(value) => Arc::new(UInt32Array::from_value(*value, size)),
                None => new_null_array(&DataType::UInt32, size),
            },
            Scalar::UInt64(e) => match e {
                Some(value) => Arc::new(UInt64Array::from_value(*value, size)),
                None => new_null_array(&DataType::UInt64, size),
            },
            Scalar::Float32(e) => match e {
                Some(value) => Arc::new(Float32Array::from_value(*value, size)),
                None => new_null_array(&DataType::Float32, size),
            },
            Scalar::Float64(e) => match e {
                Some(value) => Arc::new(Float64Array::from_value(*value, size)),
                None => new_null_array(&DataType::Float64, size),
            },
            Scalar::Utf8(e) => match e {
                Some(value) => Arc::new(StringArray::from_iter_values(repeat_n(value, size))),
                None => new_null_array(&DataType::Utf8, size),
            },
            Scalar::Utf8View(e) => match e {
                Some(value) => Arc::new(StringViewArray::from_iter_values(repeat_n(value, size))),
                None => new_null_array(&DataType::Utf8View, size),
            },
            Scalar::LargeUtf8(e) => match e {
                Some(value) => Arc::new(LargeStringArray::from_iter_values(repeat_n(value, size))),
                None => new_null_array(&DataType::LargeUtf8, size),
            },
            Scalar::Binary(e) => match e {
                Some(value) => {
                    Arc::new(repeat_n(Some(value.as_slice()), size).collect::<BinaryArray>())
                }
                None => Arc::new(repeat_n(None::<&str>, size).collect::<BinaryArray>()),
            },
            Scalar::BinaryView(e) => match e {
                Some(value) => Arc::new(BinaryViewArray::from_iter_values(repeat_n(value, size))),
                None => new_null_array(&DataType::BinaryView, size),
            },
            Scalar::FixedSizeBinary(s, e) => match e {
                Some(value) => Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        repeat_n(Some(value.as_slice()), size),
                        *s,
                    )
                    .unwrap(),
                ),
                None => Arc::new(
                    FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                        repeat_n(None::<&[u8]>, size),
                        *s,
                    )
                    .unwrap(),
                ),
            },
            Scalar::LargeBinary(e) => match e {
                Some(value) => Arc::new(LargeBinaryArray::from_iter_values(repeat_n(value, size))),
                None => new_null_array(&DataType::LargeBinary, size),
            },
            Scalar::TimestampSecond(e, tz) => match e {
                Some(value) => Arc::new(
                    TimestampSecondArray::from_value(*value, size).with_timezone_opt(tz.clone()),
                ),
                None => new_null_array(&DataType::Timestamp(TimeUnit::Second, tz.clone()), size),
            },
            Scalar::TimestampMillisecond(e, tz) => match e {
                Some(value) => Arc::new(
                    TimestampMillisecondArray::from_value(*value, size)
                        .with_timezone_opt(tz.clone()),
                ),
                None => new_null_array(
                    &DataType::Timestamp(TimeUnit::Millisecond, tz.clone()),
                    size,
                ),
            },
            Scalar::TimestampMicrosecond(e, tz) => match e {
                Some(value) => Arc::new(
                    TimestampMicrosecondArray::from_value(*value, size)
                        .with_timezone_opt(tz.clone()),
                ),
                None => new_null_array(
                    &DataType::Timestamp(TimeUnit::Microsecond, tz.clone()),
                    size,
                ),
            },
            Scalar::TimestampNanosecond(e, tz) => match e {
                Some(value) => Arc::new(
                    TimestampNanosecondArray::from_value(*value, size)
                        .with_timezone_opt(tz.clone()),
                ),
                None => {
                    new_null_array(&DataType::Timestamp(TimeUnit::Nanosecond, tz.clone()), size)
                }
            },
            Scalar::Date32(e) => match e {
                Some(value) => Arc::new(Date32Array::from_value(*value, size)),
                None => new_null_array(&DataType::Date32, size),
            },
            Scalar::Date64(e) => match e {
                Some(value) => Arc::new(Date64Array::from_value(*value, size)),
                None => new_null_array(&DataType::Date64, size),
            },
            Scalar::Time32Second(e) => match e {
                Some(value) => Arc::new(Time32SecondArray::from_value(*value, size)),
                None => new_null_array(&DataType::Time32(TimeUnit::Second), size),
            },
            Scalar::Time32Millisecond(e) => match e {
                Some(value) => Arc::new(Time32MillisecondArray::from_value(*value, size)),
                None => new_null_array(&DataType::Time32(TimeUnit::Millisecond), size),
            },
            Scalar::Time64Microsecond(e) => match e {
                Some(value) => Arc::new(Time64MicrosecondArray::from_value(*value, size)),
                None => new_null_array(&DataType::Time64(TimeUnit::Microsecond), size),
            },
            Scalar::Time64Nanosecond(e) => match e {
                Some(value) => Arc::new(Time64NanosecondArray::from_value(*value, size)),
                None => new_null_array(&DataType::Time64(TimeUnit::Nanosecond), size),
            },
            Scalar::List(arr) => {
                if size == 1 {
                    return Ok(Arc::clone(arr) as Arc<dyn Array>);
                }
                Self::list_to_array_of_size(arr.as_ref() as &dyn Array, size)?
            }
            Scalar::FixedSizeList(arr) => {
                if size == 1 {
                    return Ok(Arc::clone(arr) as Arc<dyn Array>);
                }
                Self::list_to_array_of_size(arr.as_ref() as &dyn Array, size)?
            }
            Scalar::LargeList(arr) => {
                if size == 1 {
                    return Ok(Arc::clone(arr) as Arc<dyn Array>);
                }
                Self::list_to_array_of_size(arr.as_ref() as &dyn Array, size)?
            }
            Scalar::Decimal128(e, precision, scale) => match e {
                Some(value) => Arc::new(
                    Decimal128Array::from_value(*value, size)
                        .with_precision_and_scale(*precision, *scale)?,
                ),
                None => new_null_array(&DataType::Decimal128(*precision, *scale), size),
            },
            Scalar::Decimal256(e, precision, scale) => match e {
                Some(value) => Arc::new(
                    Decimal256Array::from_value(*value, size)
                        .with_precision_and_scale(*precision, *scale)?,
                ),
                None => new_null_array(&DataType::Decimal256(*precision, *scale), size),
            },
        })
    }

    pub fn list_to_array_of_size(arr: &dyn Array, size: usize) -> ILResult<ArrayRef> {
        let arrays = repeat_n(arr, size).collect::<Vec<_>>();
        let ret = match !arrays.is_empty() {
            true => arrow::compute::concat(arrays.as_slice())?,
            false => arr.slice(0, 0),
        };
        Ok(ret)
    }

    pub fn try_from_array(array: &dyn Array, index: usize) -> ILResult<Self> {
        // handle NULL value
        if !array.is_valid(index) {
            return Self::try_new_null(array.data_type());
        }
        Ok(match array.data_type() {
            DataType::Boolean => {
                let array = array.as_boolean();
                Scalar::Boolean(Some(array.value(index)))
            }
            DataType::Int8 => {
                let array = array.as_primitive::<Int8Type>();
                Scalar::Int8(Some(array.value(index)))
            }
            DataType::Int16 => {
                let array = array.as_primitive::<Int16Type>();
                Scalar::Int16(Some(array.value(index)))
            }
            DataType::Int32 => {
                let array = array.as_primitive::<Int32Type>();
                Scalar::Int32(Some(array.value(index)))
            }
            DataType::Int64 => {
                let array = array.as_primitive::<Int64Type>();
                Scalar::Int64(Some(array.value(index)))
            }
            DataType::UInt8 => {
                let array = array.as_primitive::<UInt8Type>();
                Scalar::UInt8(Some(array.value(index)))
            }
            DataType::UInt16 => {
                let array = array.as_primitive::<UInt16Type>();
                Scalar::UInt16(Some(array.value(index)))
            }
            DataType::UInt32 => {
                let array = array.as_primitive::<UInt32Type>();
                Scalar::UInt32(Some(array.value(index)))
            }
            DataType::UInt64 => {
                let array = array.as_primitive::<UInt64Type>();
                Scalar::UInt64(Some(array.value(index)))
            }
            DataType::Float32 => {
                let array = array.as_primitive::<Float32Type>();
                Scalar::Float32(Some(array.value(index)))
            }
            DataType::Float64 => {
                let array = array.as_primitive::<Float64Type>();
                Scalar::Float64(Some(array.value(index)))
            }
            DataType::Utf8 => {
                let array = array.as_string::<i32>();
                Scalar::Utf8(Some(array.value(index).to_string()))
            }
            DataType::Utf8View => {
                let array = array.as_string_view();
                Scalar::Utf8View(Some(array.value(index).to_string()))
            }
            DataType::LargeUtf8 => {
                let array = array.as_string::<i64>();
                Scalar::LargeUtf8(Some(array.value(index).to_string()))
            }
            DataType::Binary => {
                let array = array.as_binary::<i32>();
                Scalar::Binary(Some(array.value(index).to_vec()))
            }
            DataType::BinaryView => {
                let array = array.as_binary_view();
                Scalar::BinaryView(Some(array.value(index).to_vec()))
            }
            DataType::FixedSizeBinary(size) => {
                let array = array.as_fixed_size_binary();
                Scalar::FixedSizeBinary(*size, Some(array.value(index).to_vec()))
            }
            DataType::LargeBinary => {
                let array = array.as_binary::<i64>();
                Scalar::LargeBinary(Some(array.value(index).to_vec()))
            }
            DataType::Timestamp(TimeUnit::Second, tz) => {
                let array = array.as_primitive::<TimestampSecondType>();
                Scalar::TimestampSecond(Some(array.value(index)), tz.clone())
            }
            DataType::Timestamp(TimeUnit::Millisecond, tz) => {
                let array = array.as_primitive::<TimestampMillisecondType>();
                Scalar::TimestampMillisecond(Some(array.value(index)), tz.clone())
            }
            DataType::Timestamp(TimeUnit::Microsecond, tz) => {
                let array = array.as_primitive::<TimestampMicrosecondType>();
                Scalar::TimestampMicrosecond(Some(array.value(index)), tz.clone())
            }
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                let array = array.as_primitive::<TimestampNanosecondType>();
                Scalar::TimestampNanosecond(Some(array.value(index)), tz.clone())
            }
            DataType::Date32 => {
                let array = array.as_primitive::<Date32Type>();
                Scalar::Date32(Some(array.value(index)))
            }
            DataType::Date64 => {
                let array = array.as_primitive::<Date64Type>();
                Scalar::Date64(Some(array.value(index)))
            }
            DataType::Time32(TimeUnit::Second) => {
                let array = array.as_primitive::<Time32SecondType>();
                Scalar::Time32Second(Some(array.value(index)))
            }
            DataType::Time32(TimeUnit::Millisecond) => {
                let array = array.as_primitive::<Time32MillisecondType>();
                Scalar::Time32Millisecond(Some(array.value(index)))
            }
            DataType::Time64(TimeUnit::Microsecond) => {
                let array = array.as_primitive::<Time64MicrosecondType>();
                Scalar::Time64Microsecond(Some(array.value(index)))
            }
            DataType::Time64(TimeUnit::Nanosecond) => {
                let array = array.as_primitive::<Time64NanosecondType>();
                Scalar::Time64Nanosecond(Some(array.value(index)))
            }
            DataType::List(field_ref) => {
                let array = array.as_list::<i32>();
                let ele = array.value(index);
                let offsets = OffsetBuffer::from_lengths([ele.len()]);
                let list = ListArray::new(field_ref.clone(), offsets, ele, None);
                Scalar::List(Arc::new(list))
            }
            DataType::FixedSizeList(field_ref, _) => {
                let array = array.as_fixed_size_list();
                let ele = array.value(index);
                let list_size = ele.len();
                let list = FixedSizeListArray::new(field_ref.clone(), list_size as i32, ele, None);
                Scalar::FixedSizeList(Arc::new(list))
            }
            DataType::LargeList(field_ref) => {
                let array = array.as_list::<i64>();
                let ele = array.value(index);
                let offsets = OffsetBuffer::from_lengths([ele.len()]);
                let list = LargeListArray::new(field_ref.clone(), offsets, ele, None);
                Scalar::LargeList(Arc::new(list))
            }
            DataType::Decimal128(precision, scale) => {
                let array = array.as_primitive::<Decimal128Type>();
                Scalar::Decimal128(Some(array.value(index)), *precision, *scale)
            }
            DataType::Decimal256(precision, scale) => {
                let array = array.as_primitive::<Decimal256Type>();
                Scalar::Decimal256(Some(array.value(index)), *precision, *scale)
            }
            _ => {
                return Err(ILError::not_supported(format!(
                    "Unsupported array: {}",
                    array.data_type()
                )));
            }
        })
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Scalar::Boolean(_) => DataType::Boolean,
            Scalar::Int8(_) => DataType::Int8,
            Scalar::Int16(_) => DataType::Int16,
            Scalar::Int32(_) => DataType::Int32,
            Scalar::Int64(_) => DataType::Int64,
            Scalar::UInt8(_) => DataType::UInt8,
            Scalar::UInt16(_) => DataType::UInt16,
            Scalar::UInt32(_) => DataType::UInt32,
            Scalar::UInt64(_) => DataType::UInt64,
            Scalar::Float32(_) => DataType::Float32,
            Scalar::Float64(_) => DataType::Float64,
            Scalar::Utf8(_) => DataType::Utf8,
            Scalar::Utf8View(_) => DataType::Utf8View,
            Scalar::LargeUtf8(_) => DataType::LargeUtf8,
            Scalar::Binary(_) => DataType::Binary,
            Scalar::BinaryView(_) => DataType::BinaryView,
            Scalar::FixedSizeBinary(size, _) => DataType::FixedSizeBinary(*size),
            Scalar::LargeBinary(_) => DataType::LargeBinary,
            Scalar::TimestampSecond(_, tz) => DataType::Timestamp(TimeUnit::Second, tz.clone()),
            Scalar::TimestampMillisecond(_, tz) => {
                DataType::Timestamp(TimeUnit::Millisecond, tz.clone())
            }
            Scalar::TimestampMicrosecond(_, tz) => {
                DataType::Timestamp(TimeUnit::Microsecond, tz.clone())
            }
            Scalar::TimestampNanosecond(_, tz) => {
                DataType::Timestamp(TimeUnit::Nanosecond, tz.clone())
            }
            Scalar::Date32(_) => DataType::Date32,
            Scalar::Date64(_) => DataType::Date64,
            Scalar::Time32Second(_) => DataType::Time32(TimeUnit::Second),
            Scalar::Time32Millisecond(_) => DataType::Time32(TimeUnit::Millisecond),
            Scalar::Time64Microsecond(_) => DataType::Time64(TimeUnit::Microsecond),
            Scalar::Time64Nanosecond(_) => DataType::Time64(TimeUnit::Nanosecond),
            Scalar::List(arr) => arr.data_type().clone(),
            Scalar::FixedSizeList(arr) => arr.data_type().clone(),
            Scalar::LargeList(arr) => arr.data_type().clone(),
            Scalar::Decimal128(_, precision, scale) => DataType::Decimal128(*precision, *scale),
            Scalar::Decimal256(_, precision, scale) => DataType::Decimal256(*precision, *scale),
        }
    }

    pub fn cast_to(
        &self,
        target_type: &DataType,
        cast_options: &CastOptions<'static>,
    ) -> ILResult<Self> {
        let scalar_array = self.to_array_of_size(1)?;
        let cast_arr = arrow::compute::cast_with_options(&scalar_array, target_type, cast_options)?;
        Self::try_from_array(&cast_arr, 0)
    }

    pub fn arithmetic_negate(&self) -> ILResult<Self> {
        use arrow::array::ArrowNativeTypeOp;
        match self {
            Scalar::Int8(None)
            | Scalar::Int16(None)
            | Scalar::Int32(None)
            | Scalar::Int64(None)
            | Scalar::Float32(None)
            | Scalar::Float64(None) => Ok(self.clone()),
            Scalar::Float64(Some(v)) => Ok(Scalar::Float64(Some(-v))),
            Scalar::Float32(Some(v)) => Ok(Scalar::Float32(Some(-v))),
            Scalar::Int8(Some(v)) => Ok(Scalar::Int8(Some(v.neg_checked()?))),
            Scalar::Int16(Some(v)) => Ok(Scalar::Int16(Some(v.neg_checked()?))),
            Scalar::Int32(Some(v)) => Ok(Scalar::Int32(Some(v.neg_checked()?))),
            Scalar::Int64(Some(v)) => Ok(Scalar::Int64(Some(v.neg_checked()?))),
            _ => Err(ILError::invalid_input(format!(
                "Can not run arithmetic negative on scalar value {self:?}",
            ))),
        }
    }
}

pub fn serialize_scalar(scalar: &Scalar) -> ILResult<Vec<u8>> {
    let field = Field::new("scalar", scalar.data_type(), true);
    let schema = Arc::new(Schema::new(vec![field]));
    let arr = scalar.to_array_of_size(1)?;
    let batch = RecordBatch::try_new(schema.clone(), vec![arr])?;
    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema)?;
    writer.write(&batch)?;
    writer.finish()?;
    Ok(buf)
}

pub fn deserialize_scalar(buf: &[u8]) -> ILResult<Scalar> {
    let mut reader = StreamReader::try_new(buf, None)?;
    let Some(batch) = reader.next() else {
        return Err(ILError::invalid_input(
            "Failed to deserialize scalar: empty buffer",
        ));
    };
    let batch = batch?;
    if batch.num_columns() != 1 {
        return Err(ILError::invalid_input(
            "Failed to deserialize scalar: multiple columns",
        ));
    }
    let array = batch.column(0);
    if array.len() != 1 {
        return Err(ILError::invalid_input(
            "Failed to deserialize scalar: invalid array length",
        ));
    }
    let scalar = Scalar::try_from_array(array, 0)?;
    Ok(scalar)
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Scalar::Boolean(v1), Scalar::Boolean(v2)) => v1.eq(v2),
            (Scalar::Boolean(_), _) => false,
            (Scalar::Int8(v1), Scalar::Int8(v2)) => v1.eq(v2),
            (Scalar::Int8(_), _) => false,
            (Scalar::Int16(v1), Scalar::Int16(v2)) => v1.eq(v2),
            (Scalar::Int16(_), _) => false,
            (Scalar::Int32(v1), Scalar::Int32(v2)) => v1.eq(v2),
            (Scalar::Int32(_), _) => false,
            (Scalar::Int64(v1), Scalar::Int64(v2)) => v1.eq(v2),
            (Scalar::Int64(_), _) => false,
            (Scalar::UInt8(v1), Scalar::UInt8(v2)) => v1.eq(v2),
            (Scalar::UInt8(_), _) => false,
            (Scalar::UInt16(v1), Scalar::UInt16(v2)) => v1.eq(v2),
            (Scalar::UInt16(_), _) => false,
            (Scalar::UInt32(v1), Scalar::UInt32(v2)) => v1.eq(v2),
            (Scalar::UInt32(_), _) => false,
            (Scalar::UInt64(v1), Scalar::UInt64(v2)) => v1.eq(v2),
            (Scalar::UInt64(_), _) => false,
            (Scalar::Float32(v1), Scalar::Float32(v2)) => match (v1, v2) {
                (Some(f1), Some(f2)) => f1.to_bits() == f2.to_bits(),
                _ => v1.eq(v2),
            },
            (Scalar::Float32(_), _) => false,
            (Scalar::Float64(v1), Scalar::Float64(v2)) => match (v1, v2) {
                (Some(d1), Some(d2)) => d1.to_bits() == d2.to_bits(),
                _ => v1.eq(v2),
            },
            (Scalar::Float64(_), _) => false,
            (Scalar::Utf8(v1), Scalar::Utf8(v2)) => v1.eq(v2),
            (Scalar::Utf8(_), _) => false,
            (Scalar::Utf8View(v1), Scalar::Utf8View(v2)) => v1.eq(v2),
            (Scalar::Utf8View(_), _) => false,
            (Scalar::LargeUtf8(v1), Scalar::LargeUtf8(v2)) => v1.eq(v2),
            (Scalar::LargeUtf8(_), _) => false,
            (Scalar::Binary(v1), Scalar::Binary(v2)) => v1.eq(v2),
            (Scalar::Binary(_), _) => false,
            (Scalar::BinaryView(v1), Scalar::BinaryView(v2)) => v1.eq(v2),
            (Scalar::BinaryView(_), _) => false,
            (Scalar::FixedSizeBinary(_, v1), Scalar::FixedSizeBinary(_, v2)) => v1.eq(v2),
            (Scalar::FixedSizeBinary(_, _), _) => false,
            (Scalar::LargeBinary(v1), Scalar::LargeBinary(v2)) => v1.eq(v2),
            (Scalar::LargeBinary(_), _) => false,
            (Scalar::TimestampSecond(v1, _), Scalar::TimestampSecond(v2, _)) => v1.eq(v2),
            (Scalar::TimestampSecond(_, _), _) => false,
            (Scalar::TimestampMillisecond(v1, _), Scalar::TimestampMillisecond(v2, _)) => v1.eq(v2),
            (Scalar::TimestampMillisecond(_, _), _) => false,
            (Scalar::TimestampMicrosecond(v1, _), Scalar::TimestampMicrosecond(v2, _)) => v1.eq(v2),
            (Scalar::TimestampMicrosecond(_, _), _) => false,
            (Scalar::TimestampNanosecond(v1, _), Scalar::TimestampNanosecond(v2, _)) => v1.eq(v2),
            (Scalar::TimestampNanosecond(_, _), _) => false,
            (Scalar::Date32(v1), Scalar::Date32(v2)) => v1.eq(v2),
            (Scalar::Date32(_), _) => false,
            (Scalar::Date64(v1), Scalar::Date64(v2)) => v1.eq(v2),
            (Scalar::Date64(_), _) => false,
            (Scalar::Time32Second(v1), Scalar::Time32Second(v2)) => v1.eq(v2),
            (Scalar::Time32Second(_), _) => false,
            (Scalar::Time32Millisecond(v1), Scalar::Time32Millisecond(v2)) => v1.eq(v2),
            (Scalar::Time32Millisecond(_), _) => false,
            (Scalar::Time64Microsecond(v1), Scalar::Time64Microsecond(v2)) => v1.eq(v2),
            (Scalar::Time64Microsecond(_), _) => false,
            (Scalar::Time64Nanosecond(v1), Scalar::Time64Nanosecond(v2)) => v1.eq(v2),
            (Scalar::Time64Nanosecond(_), _) => false,
            (Scalar::List(v1), Scalar::List(v2)) => v1.eq(v2),
            (Scalar::List(_), _) => false,
            (Scalar::FixedSizeList(v1), Scalar::FixedSizeList(v2)) => v1.eq(v2),
            (Scalar::FixedSizeList(_), _) => false,
            (Scalar::LargeList(v1), Scalar::LargeList(v2)) => v1.eq(v2),
            (Scalar::LargeList(_), _) => false,
            (Scalar::Decimal128(v1, p1, s1), Scalar::Decimal128(v2, p2, s2)) => {
                v1.eq(v2) && p1 == p2 && s1 == s2
            }
            (Scalar::Decimal128(_, _, _), _) => false,
            (Scalar::Decimal256(v1, p1, s1), Scalar::Decimal256(v2, p2, s2)) => {
                v1.eq(v2) && p1 == p2 && s1 == s2
            }
            (Scalar::Decimal256(_, _, _), _) => false,
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Scalar::Boolean(v1), Scalar::Boolean(v2)) => v1.partial_cmp(v2),
            (Scalar::Boolean(_), _) => None,
            (Scalar::Int8(v1), Scalar::Int8(v2)) => v1.partial_cmp(v2),
            (Scalar::Int8(_), _) => None,
            (Scalar::Int16(v1), Scalar::Int16(v2)) => v1.partial_cmp(v2),
            (Scalar::Int16(_), _) => None,
            (Scalar::Int32(v1), Scalar::Int32(v2)) => v1.partial_cmp(v2),
            (Scalar::Int32(_), _) => None,
            (Scalar::Int64(v1), Scalar::Int64(v2)) => v1.partial_cmp(v2),
            (Scalar::Int64(_), _) => None,
            (Scalar::UInt8(v1), Scalar::UInt8(v2)) => v1.partial_cmp(v2),
            (Scalar::UInt8(_), _) => None,
            (Scalar::UInt16(v1), Scalar::UInt16(v2)) => v1.partial_cmp(v2),
            (Scalar::UInt16(_), _) => None,
            (Scalar::UInt32(v1), Scalar::UInt32(v2)) => v1.partial_cmp(v2),
            (Scalar::UInt32(_), _) => None,
            (Scalar::UInt64(v1), Scalar::UInt64(v2)) => v1.partial_cmp(v2),
            (Scalar::UInt64(_), _) => None,
            (Scalar::Float32(v1), Scalar::Float32(v2)) => match (v1, v2) {
                (Some(f1), Some(f2)) => f1.partial_cmp(f2),
                _ => v1.partial_cmp(v2),
            },
            (Scalar::Float32(_), _) => None,
            (Scalar::Float64(v1), Scalar::Float64(v2)) => match (v1, v2) {
                (Some(d1), Some(d2)) => d1.partial_cmp(d2),
                _ => v1.partial_cmp(v2),
            },
            (Scalar::Float64(_), _) => None,
            (Scalar::Utf8(v1), Scalar::Utf8(v2)) => v1.partial_cmp(v2),
            (Scalar::Utf8(_), _) => None,
            (Scalar::Utf8View(v1), Scalar::Utf8View(v2)) => v1.partial_cmp(v2),
            (Scalar::Utf8View(_), _) => None,
            (Scalar::LargeUtf8(v1), Scalar::LargeUtf8(v2)) => v1.partial_cmp(v2),
            (Scalar::LargeUtf8(_), _) => None,
            (Scalar::Binary(v1), Scalar::Binary(v2)) => v1.partial_cmp(v2),
            (Scalar::Binary(_), _) => None,
            (Scalar::BinaryView(v1), Scalar::BinaryView(v2)) => v1.partial_cmp(v2),
            (Scalar::BinaryView(_), _) => None,
            (Scalar::FixedSizeBinary(_, v1), Scalar::FixedSizeBinary(_, v2)) => v1.partial_cmp(v2),
            (Scalar::FixedSizeBinary(_, _), _) => None,
            (Scalar::LargeBinary(v1), Scalar::LargeBinary(v2)) => v1.partial_cmp(v2),
            (Scalar::LargeBinary(_), _) => None,
            (Scalar::TimestampSecond(v1, _), Scalar::TimestampSecond(v2, _)) => v1.partial_cmp(v2),
            (Scalar::TimestampSecond(_, _), _) => None,
            (Scalar::TimestampMillisecond(v1, _), Scalar::TimestampMillisecond(v2, _)) => {
                v1.partial_cmp(v2)
            }
            (Scalar::TimestampMillisecond(_, _), _) => None,
            (Scalar::TimestampMicrosecond(v1, _), Scalar::TimestampMicrosecond(v2, _)) => {
                v1.partial_cmp(v2)
            }
            (Scalar::TimestampMicrosecond(_, _), _) => None,
            (Scalar::TimestampNanosecond(v1, _), Scalar::TimestampNanosecond(v2, _)) => {
                v1.partial_cmp(v2)
            }
            (Scalar::TimestampNanosecond(_, _), _) => None,
            (Scalar::Date32(v1), Scalar::Date32(v2)) => v1.partial_cmp(v2),
            (Scalar::Date32(_), _) => None,
            (Scalar::Date64(v1), Scalar::Date64(v2)) => v1.partial_cmp(v2),
            (Scalar::Date64(_), _) => None,
            (Scalar::Time32Second(v1), Scalar::Time32Second(v2)) => v1.partial_cmp(v2),
            (Scalar::Time32Second(_), _) => None,
            (Scalar::Time32Millisecond(v1), Scalar::Time32Millisecond(v2)) => v1.partial_cmp(v2),
            (Scalar::Time32Millisecond(_), _) => None,
            (Scalar::Time64Microsecond(v1), Scalar::Time64Microsecond(v2)) => v1.partial_cmp(v2),
            (Scalar::Time64Microsecond(_), _) => None,
            (Scalar::Time64Nanosecond(v1), Scalar::Time64Nanosecond(v2)) => v1.partial_cmp(v2),
            (Scalar::Time64Nanosecond(_), _) => None,
            (Scalar::List(arr1), Scalar::List(arr2)) => {
                partial_cmp_list(arr1.as_ref(), arr2.as_ref())
            }
            (Scalar::List(_), _) => None,
            (Scalar::FixedSizeList(v1), Scalar::FixedSizeList(v2)) => {
                partial_cmp_list(v1.as_ref(), v2.as_ref())
            }
            (Scalar::FixedSizeList(_), _) => None,
            (Scalar::LargeList(v1), Scalar::LargeList(v2)) => {
                partial_cmp_list(v1.as_ref(), v2.as_ref())
            }
            (Scalar::LargeList(_), _) => None,
            (Scalar::Decimal128(v1, p1, s1), Scalar::Decimal128(v2, p2, s2)) => {
                if p1.eq(p2) && s1.eq(s2) {
                    v1.partial_cmp(v2)
                } else {
                    // Two decimal values can be compared if they have the same precision and scale.
                    None
                }
            }
            (Scalar::Decimal128(_, _, _), _) => None,
            (Scalar::Decimal256(v1, p1, s1), Scalar::Decimal256(v2, p2, s2)) => {
                if p1.eq(p2) && s1.eq(s2) {
                    v1.partial_cmp(v2)
                } else {
                    // Two decimal values can be compared if they have the same precision and scale.
                    None
                }
            }
            (Scalar::Decimal256(_, _, _), _) => None,
        }
    }
}

/// Compares two List/LargeList/FixedSizeList scalars
fn partial_cmp_list(arr1: &dyn Array, arr2: &dyn Array) -> Option<Ordering> {
    if arr1.data_type() != arr2.data_type() {
        return None;
    }
    let arr1 = first_array_for_list(arr1);
    let arr2 = first_array_for_list(arr2);

    let min_length = arr1.len().min(arr2.len());
    let arr1_trimmed = arr1.slice(0, min_length);
    let arr2_trimmed = arr2.slice(0, min_length);

    let lt_res = arrow::compute::kernels::cmp::lt(&arr1_trimmed, &arr2_trimmed).ok()?;
    let eq_res = arrow::compute::kernels::cmp::eq(&arr1_trimmed, &arr2_trimmed).ok()?;

    for j in 0..lt_res.len() {
        // In Postgres, NULL values in lists are always considered to be greater than non-NULL values:
        //
        // $ SELECT ARRAY[NULL]::integer[] > ARRAY[1]
        // true
        //
        // These next two if statements are introduced for replicating Postgres behavior, as
        // arrow::compute does not account for this.
        if arr1_trimmed.is_null(j) && !arr2_trimmed.is_null(j) {
            return Some(Ordering::Greater);
        }
        if !arr1_trimmed.is_null(j) && arr2_trimmed.is_null(j) {
            return Some(Ordering::Less);
        }

        if lt_res.is_valid(j) && lt_res.value(j) {
            return Some(Ordering::Less);
        }
        if eq_res.is_valid(j) && !eq_res.value(j) {
            return Some(Ordering::Greater);
        }
    }

    Some(arr1.len().cmp(&arr2.len()))
}

/// List/LargeList/FixedSizeList scalars always have a single element
/// array. This function returns that array
fn first_array_for_list(arr: &dyn Array) -> ArrayRef {
    assert_eq!(arr.len(), 1);
    if let Some(arr) = arr.as_list_opt::<i32>() {
        arr.value(0)
    } else if let Some(arr) = arr.as_list_opt::<i64>() {
        arr.value(0)
    } else if let Some(arr) = arr.as_fixed_size_list_opt() {
        arr.value(0)
    } else {
        unreachable!(
            "Since only List / LargeList / FixedSizeList are supported, this should never happen"
        )
    }
}

macro_rules! format_option {
    ($F:expr, $EXPR:expr) => {{
        match $EXPR {
            Some(e) => write!($F, "{e}"),
            None => write!($F, "null"),
        }
    }};
}

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Boolean(v) => format_option!(f, v),
            Scalar::Int8(v) => format_option!(f, v),
            Scalar::Int16(v) => format_option!(f, v),
            Scalar::Int32(v) => format_option!(f, v),
            Scalar::Int64(v) => format_option!(f, v),
            Scalar::UInt8(v) => format_option!(f, v),
            Scalar::UInt16(v) => format_option!(f, v),
            Scalar::UInt32(v) => format_option!(f, v),
            Scalar::UInt64(v) => format_option!(f, v),
            Scalar::Float32(v) => format_option!(f, v),
            Scalar::Float64(v) => format_option!(f, v),
            Scalar::Utf8(v) | Scalar::Utf8View(v) | Scalar::LargeUtf8(v) => format_option!(f, v),
            Scalar::Binary(v)
            | Scalar::BinaryView(v)
            | Scalar::FixedSizeBinary(_, v)
            | Scalar::LargeBinary(v) => match v {
                Some(v) => write!(f, "{}", hex::encode(v)),
                None => write!(f, "null"),
            },
            Scalar::TimestampSecond(v, _)
            | Scalar::TimestampMillisecond(v, _)
            | Scalar::TimestampMicrosecond(v, _)
            | Scalar::TimestampNanosecond(v, _) => format_option!(f, v),
            Scalar::Date32(v) => format_option!(f, v),
            Scalar::Date64(v) => format_option!(f, v),
            Scalar::Time32Second(v) => format_option!(f, v),
            Scalar::Time32Millisecond(v) => format_option!(f, v),
            Scalar::Time64Microsecond(v) => format_option!(f, v),
            Scalar::Time64Nanosecond(v) => format_option!(f, v),
            Scalar::List(arr) => fmt_list(arr.to_owned() as ArrayRef, f),
            Scalar::FixedSizeList(arr) => fmt_list(arr.to_owned() as ArrayRef, f),
            Scalar::LargeList(arr) => fmt_list(arr.to_owned() as ArrayRef, f),
            Scalar::Decimal128(v, p, s) => {
                write!(f, "{v:?},{p:?},{s:?}")
            }
            Scalar::Decimal256(v, p, s) => {
                write!(f, "{v:?},{p:?},{s:?}")
            }
        }
    }
}

fn fmt_list(arr: ArrayRef, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    // ScalarValue List, LargeList, FixedSizeList should always have a single element
    assert_eq!(arr.len(), 1);
    let options = FormatOptions::default().with_display_error(true);
    let formatter = ArrayFormatter::try_new(arr.as_ref() as &dyn Array, &options).unwrap();
    let value_formatter = formatter.value(0);
    write!(f, "{value_formatter}")
}

macro_rules! impl_scalar_from {
    ($ty:ty, $scalar:tt) => {
        impl From<$ty> for Scalar {
            fn from(value: $ty) -> Self {
                Scalar::$scalar(Some(value))
            }
        }

        impl From<Option<$ty>> for Scalar {
            fn from(value: Option<$ty>) -> Self {
                Scalar::$scalar(value)
            }
        }
    };
}

impl_scalar_from!(bool, Boolean);
impl_scalar_from!(i8, Int8);
impl_scalar_from!(i16, Int16);
impl_scalar_from!(i32, Int32);
impl_scalar_from!(i64, Int64);
impl_scalar_from!(u8, UInt8);
impl_scalar_from!(u16, UInt16);
impl_scalar_from!(u32, UInt32);
impl_scalar_from!(u64, UInt64);
impl_scalar_from!(f32, Float32);
impl_scalar_from!(f64, Float64);
impl_scalar_from!(String, Utf8);
impl_scalar_from!(Vec<u8>, Binary);

impl From<&str> for Scalar {
    fn from(value: &str) -> Self {
        Scalar::Utf8(Some(value.to_string()))
    }
}
impl From<Option<&str>> for Scalar {
    fn from(value: Option<&str>) -> Self {
        Scalar::Utf8(value.map(|s| s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_new_null() {
        fn assert(data_type: &DataType) {
            let scalar = Scalar::try_new_null(data_type).unwrap();
            assert_eq!(&scalar.data_type(), data_type);
            assert!(scalar.is_null());
        }
        assert(&DataType::Boolean);
        assert(&DataType::Int8);
        assert(&DataType::Int16);
        assert(&DataType::Int32);
        assert(&DataType::Int64);
        assert(&DataType::UInt8);
        assert(&DataType::UInt16);
        assert(&DataType::UInt32);
        assert(&DataType::UInt64);
        assert(&DataType::Float32);
        assert(&DataType::Float64);
        assert(&DataType::Utf8);
        assert(&DataType::Utf8View);
        assert(&DataType::LargeUtf8);
        assert(&DataType::Binary);
        assert(&DataType::BinaryView);
        assert(&DataType::FixedSizeBinary(3));
        assert(&DataType::LargeBinary);
        assert(&DataType::Timestamp(TimeUnit::Second, None));
        assert(&DataType::Timestamp(TimeUnit::Millisecond, None));
        assert(&DataType::Timestamp(TimeUnit::Microsecond, None));
        assert(&DataType::Timestamp(TimeUnit::Nanosecond, None));
        assert(&DataType::Date32);
        assert(&DataType::Date64);
        assert(&DataType::Time32(TimeUnit::Second));
        assert(&DataType::Time32(TimeUnit::Millisecond));
        assert(&DataType::Time64(TimeUnit::Microsecond));
        assert(&DataType::Time64(TimeUnit::Nanosecond));
        assert(&DataType::List(Arc::new(Field::new(
            "item",
            DataType::Int32,
            false,
        ))));
        assert(&DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Int32, false)),
            3,
        ));
        assert(&DataType::LargeList(Arc::new(Field::new(
            "item",
            DataType::Int32,
            false,
        ))));
        assert(&DataType::Decimal128(10, 2));
        assert(&DataType::Decimal256(38, 18));
    }

    #[test]
    fn test_scalar_serialization_and_to_array() {
        fn assert(scalar: Scalar) {
            let buf = serialize_scalar(&scalar).unwrap();
            let de_scalar = deserialize_scalar(&buf).unwrap();
            assert_eq!(scalar, de_scalar);
            let array = scalar.to_array_of_size(2).unwrap();
            assert_eq!(array.len(), 2);
            assert_eq!(array.data_type(), &scalar.data_type());
            let scalar_from_array = Scalar::try_from_array(&array, 1).unwrap();
            assert_eq!(scalar_from_array, scalar);
        }
        assert(Scalar::Boolean(Some(true)));
        assert(Scalar::Boolean(None));
        assert(Scalar::Int8(Some(1i8)));
        assert(Scalar::Int8(None));
        assert(Scalar::Int16(Some(1i16)));
        assert(Scalar::Int16(None));
        assert(Scalar::Int32(Some(1i32)));
        assert(Scalar::Int32(None));
        assert(Scalar::Int64(Some(1i64)));
        assert(Scalar::Int64(None));
        assert(Scalar::UInt8(Some(1u8)));
        assert(Scalar::UInt8(None));
        assert(Scalar::UInt16(Some(1u16)));
        assert(Scalar::UInt16(None));
        assert(Scalar::UInt32(Some(1u32)));
        assert(Scalar::UInt32(None));
        assert(Scalar::UInt64(Some(1u64)));
        assert(Scalar::UInt64(None));
        assert(Scalar::Float32(Some(1f32)));
        assert(Scalar::Float32(None));
        assert(Scalar::Float64(Some(1f64)));
        assert(Scalar::Float64(None));
        assert(Scalar::Utf8(Some("test".to_string())));
        assert(Scalar::Utf8(None));
        assert(Scalar::Utf8View(Some("test".to_string())));
        assert(Scalar::Utf8View(None));
        assert(Scalar::LargeUtf8(Some("test".to_string())));
        assert(Scalar::LargeUtf8(None));
        assert(Scalar::Binary(Some(vec![1u8, 2, 3])));
        assert(Scalar::Binary(None));
        assert(Scalar::BinaryView(Some(vec![1u8, 2, 3])));
        assert(Scalar::BinaryView(None));
        assert(Scalar::FixedSizeBinary(3, Some(vec![1u8, 2, 3])));
        assert(Scalar::FixedSizeBinary(3, None));
        assert(Scalar::LargeBinary(Some(vec![1u8, 2, 3])));
        assert(Scalar::LargeBinary(None));
        assert(Scalar::TimestampSecond(Some(1), Some("utc".into())));
        assert(Scalar::TimestampSecond(None, None));
        assert(Scalar::TimestampMillisecond(Some(1), Some("utc".into())));
        assert(Scalar::TimestampMillisecond(None, None));
        assert(Scalar::TimestampMicrosecond(Some(1), Some("utc".into())));
        assert(Scalar::TimestampMicrosecond(None, None));
        assert(Scalar::TimestampNanosecond(Some(1), Some("utc".into())));
        assert(Scalar::TimestampNanosecond(None, None));
        assert(Scalar::Date32(Some(1)));
        assert(Scalar::Date32(None));
        assert(Scalar::Date64(Some(1)));
        assert(Scalar::Date64(None));
        assert(Scalar::Time32Second(Some(1)));
        assert(Scalar::Time32Second(None));
        assert(Scalar::Time32Millisecond(Some(1)));
        assert(Scalar::Time32Millisecond(None));
        assert(Scalar::Time64Microsecond(Some(1)));
        assert(Scalar::Time64Microsecond(None));
        assert(Scalar::Time64Nanosecond(Some(1)));
        assert(Scalar::Time64Nanosecond(None));
        assert(Scalar::List(Arc::new(ListArray::from_iter_primitive::<
            Int32Type,
            _,
            _,
        >(vec![Some(vec![
            Some(0),
            Some(1),
            Some(2),
        ])]))));
        assert(
            Scalar::try_new_null(&DataType::List(Arc::new(Field::new(
                "item",
                DataType::Int32,
                false,
            ))))
            .unwrap(),
        );
        assert(Scalar::FixedSizeList(Arc::new(
            FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
                vec![Some(vec![Some(0), Some(1), Some(2)])],
                3,
            ),
        )));
        assert(
            Scalar::try_new_null(&DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Int32, false)),
                3,
            ))
            .unwrap(),
        );
        assert(Scalar::LargeList(Arc::new(
            LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![Some(vec![
                Some(0),
                Some(1),
                Some(2),
            ])]),
        )));
        assert(
            Scalar::try_new_null(&DataType::LargeList(Arc::new(Field::new(
                "item",
                DataType::Int32,
                false,
            ))))
            .unwrap(),
        );
        assert(Scalar::Decimal128(Some(1i128), 10, 2));
        assert(Scalar::Decimal128(None, 10, 2));
        assert(Scalar::Decimal256(Some(i256::from_i128(1)), 38, 18));
        assert(Scalar::Decimal256(None, 38, 18));
    }
}
