use futures::StreamExt;
use indexlake::{
    ILError, ILResult,
    catalog::{Catalog, CatalogDatabase, RowStream, Transaction},
    record::{DataType, Row, Scalar, SchemaRef},
};
use log::debug;
use std::path::PathBuf;

#[derive(Debug)]
pub struct SqliteCatalog {
    path: PathBuf,
}

impl SqliteCatalog {
    pub fn try_new(path: impl Into<String>) -> ILResult<Self> {
        let path = PathBuf::from(path.into());
        if !path.exists() {
            return Err(ILError::CatalogError(format!(
                "sqlite path {} does not exist",
                path.display()
            )));
        }
        Ok(SqliteCatalog { path })
    }
}

#[async_trait::async_trait]
impl Catalog for SqliteCatalog {
    fn database(&self) -> CatalogDatabase {
        CatalogDatabase::Sqlite
    }

    async fn transaction(&self) -> ILResult<Box<dyn Transaction>> {
        let conn = rusqlite::Connection::open(&self.path)
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        conn.execute_batch("BEGIN DEFERRED")
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        Ok(Box::new(SqliteTransaction { conn, done: false }))
    }
}

#[derive(Debug)]
pub struct SqliteTransaction {
    // TODO use tokio rusqlite connection
    conn: rusqlite::Connection,
    done: bool,
}

#[async_trait::async_trait]
impl Transaction for SqliteTransaction {
    async fn query(&mut self, sql: &str, schema: SchemaRef) -> ILResult<RowStream> {
        debug!("sqlite txn query: {sql}");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        let mut sqlite_rows = stmt
            .query([])
            .map_err(|e| ILError::CatalogError(e.to_string()))?;

        let mut rows: Vec<Row> = Vec::new();
        while let Some(sqlite_row) = sqlite_rows
            .next()
            .map_err(|e| ILError::CatalogError(e.to_string()))?
        {
            let row = sqlite_row_to_row(sqlite_row, &schema)?;
            rows.push(row);
        }
        Ok(Box::pin(futures::stream::iter(rows).map(Ok)))
    }

    async fn execute(&mut self, sql: &str) -> ILResult<usize> {
        debug!("sqlite txn execute: {sql}");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .execute(sql, [])
            .map_err(|e| ILError::CatalogError(e.to_string()))
    }

    async fn execute_batch(&mut self, sqls: &[String]) -> ILResult<()> {
        debug!("sqlite txn execute batch: {:?}", sqls);
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .execute_batch(sqls.join(";").as_str())
            .map_err(|e| ILError::CatalogError(e.to_string()))
    }

    async fn commit(&mut self) -> ILResult<()> {
        debug!("sqlite txn commit");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .execute_batch("COMMIT")
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        self.done = true;
        Ok(())
    }

    async fn rollback(&mut self) -> ILResult<()> {
        debug!("sqlite txn rollback");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .execute_batch("ROLLBACK")
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        self.done = true;
        Ok(())
    }
}

impl Drop for SqliteTransaction {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        self.conn.execute_batch("ROLLBACK").unwrap();
    }
}

fn sqlite_row_to_row(sqlite_row: &rusqlite::Row, schema: &SchemaRef) -> ILResult<Row> {
    let mut row_values = Vec::new();
    for (idx, field) in schema.fields.iter().enumerate() {
        let scalar = match field.data_type {
            DataType::Int16 => {
                let v: Option<i16> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int16(v)
            }
            DataType::Int32 => {
                let v: Option<i32> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int32(v)
            }
            DataType::Int64 => {
                let v: Option<i64> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int64(v)
            }
            DataType::Float32 => {
                let v: Option<f32> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Float32(v)
            }
            DataType::Float64 => {
                let v: Option<f64> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Float64(v)
            }
            DataType::Utf8 => {
                let v: Option<String> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Utf8(v)
            }
            DataType::Binary => {
                let v: Option<Vec<u8>> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Binary(v)
            }
            DataType::Boolean => {
                let v: Option<bool> = sqlite_row
                    .get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Boolean(v)
            }
        };
        if !field.nullable && scalar.is_null() {
            return Err(ILError::CatalogError(format!(
                "column {} is not nullable but got null value",
                field.name
            )));
        }
        row_values.push(scalar);
    }
    Ok(Row::new(schema.clone(), row_values))
}
