use std::error::Error;

use arrow::{array::Array, datatypes::DataType};
use bb8_postgres::tokio_postgres::types::{IsNull, ToSql, Type};
use bytes::BytesMut;
use indexlake::{
    ILError, ILResult,
    catalog::{CatalogDataType, Scalar},
    utils::serialize_array,
};
use uuid::Uuid;

pub fn catalog_data_type_to_pg_type(data_type: CatalogDataType) -> Type {
    match data_type {
        CatalogDataType::Boolean => Type::BOOL,
        CatalogDataType::Int8 | CatalogDataType::Int16 | CatalogDataType::UInt8 => Type::INT2,
        CatalogDataType::Int32 | CatalogDataType::UInt16 => Type::INT4,
        CatalogDataType::Int64 | CatalogDataType::UInt32 => Type::INT8,
        CatalogDataType::UInt64 | CatalogDataType::Float32 => Type::FLOAT4,
        CatalogDataType::Float64 => Type::FLOAT8,
        CatalogDataType::Utf8 => Type::VARCHAR,
        CatalogDataType::Binary => Type::BYTEA,
        CatalogDataType::Uuid => Type::UUID,
    }
}

#[derive(Debug)]
pub enum PgValue {
    Bool(Option<bool>),
    Int2(Option<i16>),
    Int4(Option<i32>),
    Int8(Option<i64>),
    Float4(Option<f32>),
    Float8(Option<f64>),
    Varchar(Option<String>),
    Bytea(Option<Vec<u8>>),
    Uuid(Option<Uuid>),
}

impl ToSql for PgValue {
    fn to_sql(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        match self {
            PgValue::Bool(v) => v.to_sql(ty, out),
            PgValue::Int2(v) => v.to_sql(ty, out),
            PgValue::Int4(v) => v.to_sql(ty, out),
            PgValue::Int8(v) => v.to_sql(ty, out),
            PgValue::Float4(v) => v.to_sql(ty, out),
            PgValue::Float8(v) => v.to_sql(ty, out),
            PgValue::Varchar(v) => v.to_sql(ty, out),
            PgValue::Bytea(v) => v.to_sql(ty, out),
            PgValue::Uuid(v) => v.to_sql(ty, out),
        }
    }

    fn accepts(_ty: &Type) -> bool {
        true
    }

    fn to_sql_checked(
        &self,
        ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        match self {
            PgValue::Bool(v) => v.to_sql_checked(ty, out),
            PgValue::Int2(v) => v.to_sql_checked(ty, out),
            PgValue::Int4(v) => v.to_sql_checked(ty, out),
            PgValue::Int8(v) => v.to_sql_checked(ty, out),
            PgValue::Float4(v) => v.to_sql_checked(ty, out),
            PgValue::Float8(v) => v.to_sql_checked(ty, out),
            PgValue::Varchar(v) => v.to_sql_checked(ty, out),
            PgValue::Bytea(v) => v.to_sql_checked(ty, out),
            PgValue::Uuid(v) => v.to_sql_checked(ty, out),
        }
    }
}

pub fn scalar_to_pg_value(scalar: Scalar) -> ILResult<PgValue> {
    Ok(match scalar {
        Scalar::Boolean(v) => PgValue::Bool(v),
        Scalar::Int8(v) => PgValue::Int2(v.map(|v| v as i16)),
        Scalar::Int16(v) => PgValue::Int2(v),
        Scalar::Int32(v)
        | Scalar::Date32(v)
        | Scalar::Time32Second(v)
        | Scalar::Time32Millisecond(v) => PgValue::Int4(v),
        Scalar::Int64(v)
        | Scalar::Date64(v)
        | Scalar::Time64Microsecond(v)
        | Scalar::Time64Nanosecond(v)
        | Scalar::TimestampSecond(v, _)
        | Scalar::TimestampMillisecond(v, _)
        | Scalar::TimestampMicrosecond(v, _)
        | Scalar::TimestampNanosecond(v, _) => PgValue::Int8(v),
        Scalar::UInt8(v) => PgValue::Int2(v.map(|v| v as i16)),
        Scalar::UInt16(v) => PgValue::Int4(v.map(|v| v as i32)),
        Scalar::UInt32(v) => PgValue::Int8(v.map(|v| v as i64)),
        Scalar::UInt64(v) => PgValue::Float4(v.map(|v| v as f32)),
        Scalar::Float32(v) => PgValue::Float4(v),
        Scalar::Float64(v) => PgValue::Float8(v),
        Scalar::Decimal128(v, ..) => PgValue::Varchar(v.map(|v| v.to_string())),
        Scalar::Decimal256(v, ..) => PgValue::Varchar(v.map(|v| v.to_string())),
        Scalar::FixedSizeBinary(16, v) => {
            PgValue::Uuid(v.map(|v| Uuid::from_slice(&v)).transpose()?)
        }
        Scalar::FixedSizeBinary(_, v) => PgValue::Bytea(v),
        Scalar::Binary(v) | Scalar::LargeBinary(v) | Scalar::BinaryView(v) => PgValue::Bytea(v),
        Scalar::Utf8(v) | Scalar::LargeUtf8(v) | Scalar::Utf8View(v) => PgValue::Varchar(v),
        Scalar::List(arr) => {
            if arr.is_null(0) {
                PgValue::Bytea(None)
            } else {
                let DataType::List(f) = arr.data_type() else {
                    return Err(ILError::internal("failed to get List data type"));
                };
                PgValue::Bytea(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
        Scalar::FixedSizeList(arr) => {
            if arr.is_null(0) {
                PgValue::Bytea(None)
            } else {
                let DataType::FixedSizeList(f, _) = arr.data_type() else {
                    return Err(ILError::internal("failed to get FixedSizeList data type"));
                };
                PgValue::Bytea(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
        Scalar::LargeList(arr) => {
            if arr.is_null(0) {
                PgValue::Bytea(None)
            } else {
                let DataType::LargeList(f) = arr.data_type() else {
                    return Err(ILError::internal("failed to get LargeList data type"));
                };
                PgValue::Bytea(Some(serialize_array(arr.value(0), f.clone())?))
            }
        }
    })
}
