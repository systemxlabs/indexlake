use std::fmt::Display;

use derive_visitor::{Drive, DriveMut};

use crate::{ILError, ILResult, catalog::CatalogDatabase};

#[derive(Debug, Clone, Drive, DriveMut)]
pub enum Scalar {
    Int16(Option<i16>),
    Int32(Option<i32>),
    Int64(Option<i64>),
    Float32(Option<f32>),
    Float64(Option<f64>),
    Utf8(Option<String>),
    Binary(Option<Vec<u8>>),
    Boolean(Option<bool>),
}

impl Scalar {
    pub fn is_null(&self) -> bool {
        match self {
            Scalar::Int16(None) => true,
            Scalar::Int16(Some(_)) => false,
            Scalar::Int32(None) => true,
            Scalar::Int32(Some(_)) => false,
            Scalar::Int64(None) => true,
            Scalar::Int64(Some(_)) => false,
            Scalar::Float32(None) => true,
            Scalar::Float32(Some(_)) => false,
            Scalar::Float64(None) => true,
            Scalar::Float64(Some(_)) => false,
            Scalar::Utf8(None) => true,
            Scalar::Utf8(Some(_)) => false,
            Scalar::Binary(None) => true,
            Scalar::Binary(Some(_)) => false,
            Scalar::Boolean(None) => true,
            Scalar::Boolean(Some(_)) => false,
        }
    }

    pub fn is_true(&self) -> bool {
        match self {
            Scalar::Boolean(Some(true)) => true,
            _ => false,
        }
    }

    pub fn to_sql(&self, database: CatalogDatabase) -> String {
        match self {
            Scalar::Int16(Some(value)) => value.to_string(),
            Scalar::Int16(None) => "null".to_string(),
            Scalar::Int32(Some(value)) => value.to_string(),
            Scalar::Int32(None) => "null".to_string(),
            Scalar::Int64(Some(value)) => value.to_string(),
            Scalar::Int64(None) => "null".to_string(),
            Scalar::Float32(Some(value)) => value.to_string(),
            Scalar::Float32(None) => "null".to_string(),
            Scalar::Float64(Some(value)) => value.to_string(),
            Scalar::Float64(None) => "null".to_string(),
            Scalar::Utf8(Some(value)) => format!("'{}'", value),
            Scalar::Utf8(None) => "null".to_string(),
            Scalar::Binary(Some(value)) => match database {
                CatalogDatabase::Sqlite => format!("X'{}'", hex::encode(value)),
                CatalogDatabase::Postgres => format!("E'\\\\x{}'", hex::encode(value)),
            },
            Scalar::Binary(None) => "null".to_string(),
            Scalar::Boolean(Some(value)) => value.to_string(),
            Scalar::Boolean(None) => "null".to_string(),
        }
    }

    pub fn add(&self, other: &Scalar) -> ILResult<Scalar> {
        match (self, other) {
            (Scalar::Int16(Some(v1)), Scalar::Int16(Some(v2))) => Ok(Scalar::Int16(Some(v1 + v2))),
            (Scalar::Int16(None), _) | (_, Scalar::Int16(None)) => Ok(Scalar::Int16(None)),
            (Scalar::Int32(Some(v1)), Scalar::Int32(Some(v2))) => Ok(Scalar::Int32(Some(v1 + v2))),
            (Scalar::Int32(None), _) | (_, Scalar::Int32(None)) => Ok(Scalar::Int32(None)),
            (Scalar::Int64(Some(v1)), Scalar::Int64(Some(v2))) => Ok(Scalar::Int64(Some(v1 + v2))),
            (Scalar::Int64(None), _) | (_, Scalar::Int64(None)) => Ok(Scalar::Int64(None)),
            (Scalar::Float32(Some(v1)), Scalar::Float32(Some(v2))) => {
                Ok(Scalar::Float32(Some(v1 + v2)))
            }
            (Scalar::Float32(None), _) | (_, Scalar::Float32(None)) => Ok(Scalar::Float32(None)),
            (Scalar::Float64(Some(v1)), Scalar::Float64(Some(v2))) => {
                Ok(Scalar::Float64(Some(v1 + v2)))
            }
            (Scalar::Float64(None), _) | (_, Scalar::Float64(None)) => Ok(Scalar::Float64(None)),
            _ => Err(ILError::InvalidInput(format!(
                "Cannot add scalars: {:?} and {:?}",
                self, other
            ))),
        }
    }

    pub fn sub(&self, other: &Scalar) -> ILResult<Scalar> {
        match (self, other) {
            (Scalar::Int16(Some(v1)), Scalar::Int16(Some(v2))) => Ok(Scalar::Int16(Some(v1 - v2))),
            (Scalar::Int16(None), _) | (_, Scalar::Int16(None)) => Ok(Scalar::Int16(None)),
            (Scalar::Int32(Some(v1)), Scalar::Int32(Some(v2))) => Ok(Scalar::Int32(Some(v1 - v2))),
            (Scalar::Int32(None), _) | (_, Scalar::Int32(None)) => Ok(Scalar::Int32(None)),
            (Scalar::Int64(Some(v1)), Scalar::Int64(Some(v2))) => Ok(Scalar::Int64(Some(v1 - v2))),
            (Scalar::Int64(None), _) | (_, Scalar::Int64(None)) => Ok(Scalar::Int64(None)),
            (Scalar::Float32(Some(v1)), Scalar::Float32(Some(v2))) => {
                Ok(Scalar::Float32(Some(v1 - v2)))
            }
            (Scalar::Float32(None), _) | (_, Scalar::Float32(None)) => Ok(Scalar::Float32(None)),
            (Scalar::Float64(Some(v1)), Scalar::Float64(Some(v2))) => {
                Ok(Scalar::Float64(Some(v1 - v2)))
            }
            (Scalar::Float64(None), _) | (_, Scalar::Float64(None)) => Ok(Scalar::Float64(None)),
            _ => Err(ILError::InvalidInput(format!(
                "Cannot subtract scalars: {:?} and {:?}",
                self, other
            ))),
        }
    }

    pub fn mul(&self, other: &Scalar) -> ILResult<Scalar> {
        match (self, other) {
            (Scalar::Int16(Some(v1)), Scalar::Int16(Some(v2))) => Ok(Scalar::Int16(Some(v1 * v2))),
            (Scalar::Int16(None), _) | (_, Scalar::Int16(None)) => Ok(Scalar::Int16(None)),
            (Scalar::Int32(Some(v1)), Scalar::Int32(Some(v2))) => Ok(Scalar::Int32(Some(v1 * v2))),
            (Scalar::Int32(None), _) | (_, Scalar::Int32(None)) => Ok(Scalar::Int32(None)),
            (Scalar::Int64(Some(v1)), Scalar::Int64(Some(v2))) => Ok(Scalar::Int64(Some(v1 * v2))),
            (Scalar::Int64(None), _) | (_, Scalar::Int64(None)) => Ok(Scalar::Int64(None)),
            (Scalar::Float32(Some(v1)), Scalar::Float32(Some(v2))) => {
                Ok(Scalar::Float32(Some(v1 * v2)))
            }
            (Scalar::Float32(None), _) | (_, Scalar::Float32(None)) => Ok(Scalar::Float32(None)),
            (Scalar::Float64(Some(v1)), Scalar::Float64(Some(v2))) => {
                Ok(Scalar::Float64(Some(v1 * v2)))
            }
            (Scalar::Float64(None), _) | (_, Scalar::Float64(None)) => Ok(Scalar::Float64(None)),
            _ => Err(ILError::InvalidInput(format!(
                "Cannot multiply scalars: {:?} and {:?}",
                self, other
            ))),
        }
    }

    pub fn div(&self, other: &Scalar) -> ILResult<Scalar> {
        match (self, other) {
            (Scalar::Int16(Some(v1)), Scalar::Int16(Some(v2))) => Ok(Scalar::Int16(Some(v1 / v2))),
            (Scalar::Int16(None), _) | (_, Scalar::Int16(None)) => Ok(Scalar::Int16(None)),
            (Scalar::Int32(Some(v1)), Scalar::Int32(Some(v2))) => Ok(Scalar::Int32(Some(v1 / v2))),
            (Scalar::Int32(None), _) | (_, Scalar::Int32(None)) => Ok(Scalar::Int32(None)),
            (Scalar::Int64(Some(v1)), Scalar::Int64(Some(v2))) => Ok(Scalar::Int64(Some(v1 / v2))),
            (Scalar::Int64(None), _) | (_, Scalar::Int64(None)) => Ok(Scalar::Int64(None)),
            (Scalar::Float32(Some(v1)), Scalar::Float32(Some(v2))) => {
                Ok(Scalar::Float32(Some(v1 / v2)))
            }
            (Scalar::Float32(None), _) | (_, Scalar::Float32(None)) => Ok(Scalar::Float32(None)),
            (Scalar::Float64(Some(v1)), Scalar::Float64(Some(v2))) => {
                Ok(Scalar::Float64(Some(v1 / v2)))
            }
            (Scalar::Float64(None), _) | (_, Scalar::Float64(None)) => Ok(Scalar::Float64(None)),
            _ => Err(ILError::InvalidInput(format!(
                "Cannot divide scalars: {:?} and {:?}",
                self, other
            ))),
        }
    }

    pub fn rem(&self, other: &Scalar) -> ILResult<Scalar> {
        match (self, other) {
            (Scalar::Int16(Some(v1)), Scalar::Int16(Some(v2))) => Ok(Scalar::Int16(Some(v1 % v2))),
            (Scalar::Int16(None), _) | (_, Scalar::Int16(None)) => Ok(Scalar::Int16(None)),
            (Scalar::Int32(Some(v1)), Scalar::Int32(Some(v2))) => Ok(Scalar::Int32(Some(v1 % v2))),
            (Scalar::Int32(None), _) | (_, Scalar::Int32(None)) => Ok(Scalar::Int32(None)),
            (Scalar::Int64(Some(v1)), Scalar::Int64(Some(v2))) => Ok(Scalar::Int64(Some(v1 % v2))),
            (Scalar::Int64(None), _) | (_, Scalar::Int64(None)) => Ok(Scalar::Int64(None)),
            (Scalar::Float32(Some(v1)), Scalar::Float32(Some(v2))) => {
                Ok(Scalar::Float32(Some(v1 % v2)))
            }
            (Scalar::Float32(None), _) | (_, Scalar::Float32(None)) => Ok(Scalar::Float32(None)),
            (Scalar::Float64(Some(v1)), Scalar::Float64(Some(v2))) => {
                Ok(Scalar::Float64(Some(v1 % v2)))
            }
            (Scalar::Float64(None), _) | (_, Scalar::Float64(None)) => Ok(Scalar::Float64(None)),
            _ => Err(ILError::InvalidInput(format!(
                "Cannot modulo scalars: {:?} and {:?}",
                self, other
            ))),
        }
    }
}

impl PartialEq for Scalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Scalar::Int16(v1), Scalar::Int16(v2)) => v1.eq(v2),
            (Scalar::Int16(_), _) => false,
            (Scalar::Int32(v1), Scalar::Int32(v2)) => v1.eq(v2),
            (Scalar::Int32(_), _) => false,
            (Scalar::Int64(v1), Scalar::Int64(v2)) => v1.eq(v2),
            (Scalar::Int64(_), _) => false,
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
            (Scalar::Binary(v1), Scalar::Binary(v2)) => v1.eq(v2),
            (Scalar::Binary(_), _) => false,
            (Scalar::Boolean(v1), Scalar::Boolean(v2)) => v1.eq(v2),
            (Scalar::Boolean(_), _) => false,
        }
    }
}

impl Eq for Scalar {}

impl PartialOrd for Scalar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Scalar::Int16(v1), Scalar::Int16(v2)) => v1.partial_cmp(v2),
            (Scalar::Int16(_), _) => None,
            (Scalar::Int32(v1), Scalar::Int32(v2)) => v1.partial_cmp(v2),
            (Scalar::Int32(_), _) => None,
            (Scalar::Int64(v1), Scalar::Int64(v2)) => v1.partial_cmp(v2),
            (Scalar::Int64(_), _) => None,
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
            (Scalar::Binary(v1), Scalar::Binary(v2)) => v1.partial_cmp(v2),
            (Scalar::Binary(_), _) => None,
            (Scalar::Boolean(v1), Scalar::Boolean(v2)) => v1.partial_cmp(v2),
            (Scalar::Boolean(_), _) => None,
        }
    }
}

impl Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Scalar::Int16(Some(value)) => write!(f, "{}", value),
            Scalar::Int16(None) => write!(f, "null"),
            Scalar::Int32(Some(value)) => write!(f, "{}", value),
            Scalar::Int32(None) => write!(f, "null"),
            Scalar::Int64(Some(value)) => write!(f, "{}", value),
            Scalar::Int64(None) => write!(f, "null"),
            Scalar::Float32(Some(value)) => write!(f, "{}", value),
            Scalar::Float32(None) => write!(f, "null"),
            Scalar::Float64(Some(value)) => write!(f, "{}", value),
            Scalar::Float64(None) => write!(f, "null"),
            Scalar::Utf8(Some(value)) => write!(f, "{}", value),
            Scalar::Utf8(None) => write!(f, "null"),
            Scalar::Binary(Some(value)) => write!(f, "{}", hex::encode(value)),
            Scalar::Binary(None) => write!(f, "null"),
            Scalar::Boolean(Some(value)) => write!(f, "{}", value),
            Scalar::Boolean(None) => write!(f, "null"),
        }
    }
}
