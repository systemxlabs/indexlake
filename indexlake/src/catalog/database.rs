use arrow_schema::Schema;
use uuid::Uuid;

use crate::{
    ILResult,
    catalog::Scalar,
    expr::{BinaryExpr, Expr},
};

#[derive(Debug, Clone, Copy)]
pub enum CatalogDatabase {
    Sqlite,
    Postgres,
}

impl CatalogDatabase {
    pub fn sql_identifier(&self, ident: &str) -> String {
        match self {
            CatalogDatabase::Sqlite => format!("`{ident}`"),
            CatalogDatabase::Postgres => format!("\"{ident}\""),
        }
    }

    pub fn sql_binary_literal(&self, value: &[u8]) -> String {
        match self {
            CatalogDatabase::Sqlite => format!("X'{}'", hex::encode(value)),
            CatalogDatabase::Postgres => format!("E'\\\\x{}'", hex::encode(value)),
        }
    }

    pub fn sql_uuid_literal(&self, value: &Uuid) -> String {
        match self {
            CatalogDatabase::Sqlite => self.sql_binary_literal(value.as_bytes()),
            CatalogDatabase::Postgres => format!("'{value}'"),
        }
    }

    pub fn sql_string_literal(&self, value: &str) -> String {
        let value = value.replace("'", "''");
        match self {
            CatalogDatabase::Sqlite => format!("'{value}'"),
            CatalogDatabase::Postgres => format!("'{value}'"),
        }
    }

    // TODO implement this
    pub fn supports_filter(&self, filter: &Expr, _schema: &Schema) -> ILResult<bool> {
        match filter {
            Expr::Function(_) => Ok(false),
            Expr::BinaryExpr(BinaryExpr { left, right, .. }) => {
                if let Expr::Column(_) = left.as_ref()
                    && let Expr::Literal(lit) = right.as_ref()
                    && matches!(lit.value, Scalar::List(_))
                {
                    Ok(false)
                } else if let Expr::Literal(lit) = left.as_ref()
                    && let Expr::Column(_) = right.as_ref()
                    && matches!(lit.value, Scalar::List(_))
                {
                    Ok(false)
                } else {
                    Ok(true)
                }
            }
            _ => Ok(true),
        }
    }
}

impl std::fmt::Display for CatalogDatabase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CatalogDatabase::Sqlite => write!(f, "SQLite"),
            CatalogDatabase::Postgres => write!(f, "Postgres"),
        }
    }
}
