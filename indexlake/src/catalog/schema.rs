use std::sync::Arc;

use arrow::datatypes::{DataType, Schema};
use uuid::Uuid;

use crate::catalog::CatalogDatabase;
use crate::{ILError, ILResult};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CatalogDataType {
    Boolean,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Utf8,
    Binary,
    Uuid,
}

impl CatalogDataType {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> &str {
        match self {
            CatalogDataType::Boolean => "BOOLEAN",
            CatalogDataType::Int8 => match database {
                CatalogDatabase::Sqlite => "TINYINT",
                CatalogDatabase::Postgres => "SMALLINT",
            },
            CatalogDataType::Int16 => "SMALLINT",
            CatalogDataType::Int32 => "INTEGER",
            CatalogDataType::Int64 => "BIGINT",
            CatalogDataType::UInt8 => match database {
                CatalogDatabase::Sqlite => "TINYINT UNSIGNED",
                CatalogDatabase::Postgres => "SMALLINT",
            },
            CatalogDataType::UInt16 => match database {
                CatalogDatabase::Sqlite => "SMALLINT UNSIGNED",
                CatalogDatabase::Postgres => "INTEGER",
            },
            CatalogDataType::UInt32 => match database {
                CatalogDatabase::Sqlite => "INTEGER UNSIGNED",
                CatalogDatabase::Postgres => "BIGINT",
            },
            CatalogDataType::UInt64 => match database {
                CatalogDatabase::Sqlite => "BIGINT UNSIGNED",
                CatalogDatabase::Postgres => "FLOAT4",
            },
            CatalogDataType::Float32 => match database {
                CatalogDatabase::Sqlite => "FLOAT",
                CatalogDatabase::Postgres => "FLOAT4",
            },
            CatalogDataType::Float64 => match database {
                CatalogDatabase::Sqlite => "DOUBLE",
                CatalogDatabase::Postgres => "FLOAT8",
            },
            CatalogDataType::Utf8 => "VARCHAR",
            CatalogDataType::Binary => match database {
                CatalogDatabase::Sqlite => "BLOB",
                CatalogDatabase::Postgres => "BYTEA",
            },
            CatalogDataType::Uuid => match database {
                CatalogDatabase::Sqlite => "BLOB",
                CatalogDatabase::Postgres => "UUID",
            },
        }
    }

    pub(crate) fn from_arrow(datatype: &DataType) -> ILResult<Self> {
        match datatype {
            DataType::Boolean => Ok(CatalogDataType::Boolean),
            DataType::Int8 => Ok(CatalogDataType::Int8),
            DataType::Int16 => Ok(CatalogDataType::Int16),
            DataType::Int32 => Ok(CatalogDataType::Int32),
            DataType::Int64 => Ok(CatalogDataType::Int64),
            DataType::UInt8 => Ok(CatalogDataType::UInt8),
            DataType::UInt16 => Ok(CatalogDataType::UInt16),
            DataType::UInt32 => Ok(CatalogDataType::UInt32),
            DataType::UInt64 => Ok(CatalogDataType::UInt64),
            DataType::Float32 => Ok(CatalogDataType::Float32),
            DataType::Float64 => Ok(CatalogDataType::Float64),
            DataType::Timestamp(..) => Ok(CatalogDataType::Int64),
            DataType::Date32 => Ok(CatalogDataType::Int32),
            DataType::Date64 => Ok(CatalogDataType::Int64),
            DataType::Time32(_) => Ok(CatalogDataType::Int32),
            DataType::Time64(_) => Ok(CatalogDataType::Int64),
            DataType::Binary => Ok(CatalogDataType::Binary),
            DataType::FixedSizeBinary(size) => {
                if *size == 16 {
                    Ok(CatalogDataType::Uuid)
                } else {
                    Ok(CatalogDataType::Binary)
                }
            }
            DataType::LargeBinary => Ok(CatalogDataType::Binary),
            DataType::BinaryView => Ok(CatalogDataType::Binary),
            DataType::Utf8 => Ok(CatalogDataType::Utf8),
            DataType::LargeUtf8 => Ok(CatalogDataType::Utf8),
            DataType::Utf8View => Ok(CatalogDataType::Utf8),
            DataType::List(_) => Ok(CatalogDataType::Binary),
            DataType::ListView(_) => Ok(CatalogDataType::Binary),
            DataType::FixedSizeList(..) => Ok(CatalogDataType::Binary),
            DataType::LargeList(_) => Ok(CatalogDataType::Binary),
            DataType::LargeListView(_) => Ok(CatalogDataType::Binary),
            DataType::Decimal128(..) => Ok(CatalogDataType::Utf8),
            DataType::Decimal256(..) => Ok(CatalogDataType::Utf8),
            _ => Err(ILError::not_supported(format!(
                "Unsupported datatype: {datatype}"
            ))),
        }
    }
}

impl std::fmt::Display for CatalogDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogDataType::Boolean => write!(f, "Boolean"),
            CatalogDataType::Int8 => write!(f, "Int8"),
            CatalogDataType::Int16 => write!(f, "Int16"),
            CatalogDataType::Int32 => write!(f, "Int32"),
            CatalogDataType::Int64 => write!(f, "Int64"),
            CatalogDataType::UInt8 => write!(f, "UInt8"),
            CatalogDataType::UInt16 => write!(f, "UInt16"),
            CatalogDataType::UInt32 => write!(f, "UInt32"),
            CatalogDataType::UInt64 => write!(f, "UInt64"),
            CatalogDataType::Float32 => write!(f, "Float32"),
            CatalogDataType::Float64 => write!(f, "Float64"),
            CatalogDataType::Utf8 => write!(f, "Utf8"),
            CatalogDataType::Binary => write!(f, "Binary"),
            CatalogDataType::Uuid => write!(f, "Uuid"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Column {
    pub name: String,
    pub data_type: CatalogDataType,
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, data_type: CatalogDataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            data_type,
            nullable,
        }
    }
}

pub type CatalogSchemaRef = Arc<CatalogSchema>;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CatalogSchema {
    pub columns: Vec<Column>,
}

impl CatalogSchema {
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    pub fn from_arrow(schema: &Schema) -> ILResult<Self> {
        let mut columns = Vec::with_capacity(schema.fields.len());
        for field in schema.fields.iter() {
            let catalog_datatype = CatalogDataType::from_arrow(field.data_type())?;
            columns.push(Column::new(
                field.name().clone(),
                catalog_datatype,
                field.is_nullable(),
            ));
        }
        Ok(CatalogSchema::new(columns))
    }

    pub fn index_of(&self, field_name: &str) -> Option<usize> {
        self.columns.iter().position(|f| f.name == field_name)
    }

    pub fn get_field_by_name(&self, field_name: &str) -> Option<&Column> {
        self.columns.iter().find(|f| f.name == field_name)
    }

    pub fn select_items(&self, database: CatalogDatabase) -> Vec<String> {
        self.columns
            .iter()
            .map(|f| database.sql_identifier(&f.name))
            .collect::<Vec<_>>()
    }

    pub fn placeholder_row_sql_values(
        &self,
        database: CatalogDatabase,
        num_rows: usize,
    ) -> Vec<Vec<String>> {
        let mut columns = Vec::with_capacity(self.columns.len());
        for col in self.columns.iter() {
            let placeholder = if col.nullable {
                "NULL".to_string()
            } else {
                match col.data_type {
                    CatalogDataType::Boolean => "FALSE".to_string(),
                    CatalogDataType::Int8 => "0".to_string(),
                    CatalogDataType::Int16 => "0".to_string(),
                    CatalogDataType::Int32 => "0".to_string(),
                    CatalogDataType::Int64 => "0".to_string(),
                    CatalogDataType::UInt8 => "0".to_string(),
                    CatalogDataType::UInt16 => "0".to_string(),
                    CatalogDataType::UInt32 => "0".to_string(),
                    CatalogDataType::UInt64 => "0".to_string(),
                    CatalogDataType::Float32 => "0.0".to_string(),
                    CatalogDataType::Float64 => "0.0".to_string(),
                    CatalogDataType::Utf8 => "''".to_string(),
                    CatalogDataType::Binary => database.sql_binary_literal(&[0u8]),
                    CatalogDataType::Uuid => database.sql_uuid_literal(&Uuid::nil()),
                }
            };

            columns.push(vec![placeholder; num_rows]);
        }
        columns
    }
}
