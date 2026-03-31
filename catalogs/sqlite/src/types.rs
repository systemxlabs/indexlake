use arrow::{array::Array, datatypes::DataType};
use indexlake::{ILError, ILResult, catalog::Scalar, utils::serialize_array};
use rusqlite::Result;
use rusqlite::{ToSql, types::ToSqlOutput};

pub enum SqliteParam {
    Bool(Option<bool>),
    I32(Option<i32>),
    I64(Option<i64>),
    F64(Option<f64>),
    Text(Option<String>),
    Blob(Option<Vec<u8>>),
}

impl ToSql for SqliteParam {
    fn to_sql(&self) -> Result<ToSqlOutput<'_>> {
        match self {
            SqliteParam::Bool(v) => v.to_sql(),
            SqliteParam::I32(v) => v.to_sql(),
            SqliteParam::I64(v) => v.to_sql(),
            SqliteParam::F64(v) => v.to_sql(),
            SqliteParam::Text(v) => v.to_sql(),
            SqliteParam::Blob(v) => v.to_sql(),
        }
    }
}

pub fn scalar_to_sqlite_param(scalar: Scalar) -> ILResult<SqliteParam> {
    Ok(match scalar {
        Scalar::Boolean(v) => SqliteParam::Bool(v),
        Scalar::Int8(v) => SqliteParam::I32(v.map(|v| v as i32)),
        Scalar::Int16(v) => SqliteParam::I32(v.map(|v| v as i32)),
        Scalar::Int32(v)
        | Scalar::Date32(v)
        | Scalar::Time32Second(v)
        | Scalar::Time32Millisecond(v) => SqliteParam::I32(v),
        Scalar::Int64(v)
        | Scalar::Date64(v)
        | Scalar::Time64Microsecond(v)
        | Scalar::Time64Nanosecond(v)
        | Scalar::TimestampSecond(v, _)
        | Scalar::TimestampMillisecond(v, _)
        | Scalar::TimestampMicrosecond(v, _)
        | Scalar::TimestampNanosecond(v, _) => SqliteParam::I64(v),
        Scalar::UInt8(v) => SqliteParam::I32(v.map(|v| v as i32)),
        Scalar::UInt16(v) => SqliteParam::I32(v.map(|v| v as i32)),
        Scalar::UInt32(v) => SqliteParam::I64(v.map(|v| v as i64)),
        Scalar::UInt64(v) => SqliteParam::F64(v.map(|v| v as f64)),
        Scalar::Float32(v) => SqliteParam::F64(v.map(|v| v as f64)),
        Scalar::Float64(v) => SqliteParam::F64(v),
        Scalar::Decimal128(v, ..) => SqliteParam::Text(v.map(|v| v.to_string())),
        Scalar::Decimal256(v, ..) => SqliteParam::Text(v.map(|v| v.to_string())),
        Scalar::FixedSizeBinary(_, v) => SqliteParam::Blob(v),
        Scalar::Binary(v) | Scalar::LargeBinary(v) | Scalar::BinaryView(v) => SqliteParam::Blob(v),
        Scalar::Utf8(v) | Scalar::LargeUtf8(v) | Scalar::Utf8View(v) => SqliteParam::Text(v),
        Scalar::List(arr) => {
            if arr.is_null(0) {
                SqliteParam::Blob(None)
            } else {
                let DataType::List(f) = arr.data_type() else {
                    return Err(ILError::internal("failed to get List data type"));
                };
                SqliteParam::Blob(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
        Scalar::FixedSizeList(arr) => {
            if arr.is_null(0) {
                SqliteParam::Blob(None)
            } else {
                let DataType::FixedSizeList(f, _) = arr.data_type() else {
                    return Err(ILError::internal("failed to get FixedSizeList data type"));
                };
                SqliteParam::Blob(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
        Scalar::LargeList(arr) => {
            if arr.is_null(0) {
                SqliteParam::Blob(None)
            } else {
                let DataType::LargeList(f) = arr.data_type() else {
                    return Err(ILError::internal("failed to get LargeList data type"));
                };
                SqliteParam::Blob(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
    })
}
