use crate::{ILError, ILResult, catalog::CatalogDatabase};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Int16,
    Int32,
    Int64,
    Float32,
    Float64,
    Utf8,
    Binary,
    Boolean,
}

impl DataType {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> String {
        match self {
            DataType::Int16 => "SMALLINT".to_string(),
            DataType::Int32 => "INTEGER".to_string(),
            DataType::Int64 => "BIGINT".to_string(),
            DataType::Float32 => match database {
                CatalogDatabase::Sqlite => "FLOAT".to_string(),
                CatalogDatabase::Postgres => "FLOAT4".to_string(),
            },
            DataType::Float64 => match database {
                CatalogDatabase::Sqlite => "DOUBLE".to_string(),
                CatalogDatabase::Postgres => "FLOAT8".to_string(),
            },
            DataType::Utf8 => "VARCHAR".to_string(),
            DataType::Binary => match database {
                CatalogDatabase::Sqlite => "BLOB".to_string(),
                CatalogDatabase::Postgres => "BYTEA".to_string(),
            },
            DataType::Boolean => "BOOLEAN".to_string(),
        }
    }

    pub(crate) fn parse_sql_type(s: &str, database: CatalogDatabase) -> ILResult<Self> {
        match (s, database) {
            ("SMALLINT", CatalogDatabase::Sqlite) => Ok(DataType::Int16),
            ("INTEGER", CatalogDatabase::Sqlite) => Ok(DataType::Int32),
            ("BIGINT", CatalogDatabase::Sqlite) => Ok(DataType::Int64),
            ("FLOAT", CatalogDatabase::Sqlite) => Ok(DataType::Float32),
            ("DOUBLE", CatalogDatabase::Sqlite) => Ok(DataType::Float64),
            ("VARCHAR", CatalogDatabase::Sqlite) => Ok(DataType::Utf8),
            ("BLOB", CatalogDatabase::Sqlite) => Ok(DataType::Binary),
            ("BOOLEAN", CatalogDatabase::Sqlite) => Ok(DataType::Boolean),
            ("SMALLINT", CatalogDatabase::Postgres) => Ok(DataType::Int16),
            ("INTEGER", CatalogDatabase::Postgres) => Ok(DataType::Int32),
            ("BIGINT", CatalogDatabase::Postgres) => Ok(DataType::Int64),
            ("FLOAT4", CatalogDatabase::Postgres) => Ok(DataType::Float32),
            ("FLOAT8", CatalogDatabase::Postgres) => Ok(DataType::Float64),
            ("VARCHAR", CatalogDatabase::Postgres) => Ok(DataType::Utf8),
            ("BYTEA", CatalogDatabase::Postgres) => Ok(DataType::Binary),
            ("BOOLEAN", CatalogDatabase::Postgres) => Ok(DataType::Boolean),
            _ => Err(ILError::NotSupported(format!(
                "Unsupported data type: {s} for database: {database}"
            ))),
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Int16 => write!(f, "Int16"),
            DataType::Int32 => write!(f, "Int32"),
            DataType::Int64 => write!(f, "Int64"),
            DataType::Float32 => write!(f, "Float32"),
            DataType::Float64 => write!(f, "Float64"),
            DataType::Utf8 => write!(f, "Utf8"),
            DataType::Binary => write!(f, "Binary"),
            DataType::Boolean => write!(f, "Boolean"),
        }
    }
}
