use futures::StreamExt;
use indexlake::catalog::{
    Catalog, CatalogDataType, CatalogDatabase, CatalogSchemaRef, Row, RowStream, Scalar,
    Transaction,
};
use indexlake::{ILError, ILResult};
use log::{error, trace};
use rusqlite::OpenFlags;
use std::path::PathBuf;

#[derive(Debug)]
pub struct SqliteCatalog {
    path: PathBuf,
}

impl SqliteCatalog {
    pub fn try_new(path: impl Into<String>) -> ILResult<Self> {
        let path = PathBuf::from(path.into());
        if !path.exists() {
            return Err(ILError::catalog(format!(
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

    async fn query(&self, sql: &str, schema: CatalogSchemaRef) -> ILResult<RowStream<'static>> {
        trace!("sqlite query: {sql}");
        let conn = rusqlite::Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_ONLY
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| ILError::catalog(format!("failed to open sqlite db: {e}")))?;
        let mut stmt = conn
            .prepare(sql)
            .map_err(|e| ILError::catalog(format!("failed to prepare sqlite stmt: {e}")))?;
        let mut sqlite_rows = stmt
            .query([])
            .map_err(|e| ILError::catalog(format!("failed to query sqlite stmt: {e}")))?;

        let mut rows: Vec<Row> = Vec::new();
        while let Some(sqlite_row) = sqlite_rows
            .next()
            .map_err(|e| ILError::catalog(format!("failed to get next sqlite row: {e}")))?
        {
            let row = sqlite_row_to_row(sqlite_row, &schema)?;
            rows.push(row);
        }
        Ok(Box::pin(futures::stream::iter(rows).map(Ok)))
    }

    async fn transaction(&self) -> ILResult<Box<dyn Transaction>> {
        let conn = rusqlite::Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| ILError::catalog(format!("failed to open sqlite db: {e}")))?;
        conn.execute_batch("BEGIN DEFERRED")
            .map_err(|e| ILError::catalog(format!("failed to begin sqlite txn: {e}")))?;
        Ok(Box::new(SqliteTransaction { conn, done: false }))
    }
}

#[derive(Debug)]
pub struct SqliteTransaction {
    conn: rusqlite::Connection,
    done: bool,
}

impl SqliteTransaction {
    fn check_done(&self) -> ILResult<()> {
        if self.done {
            return Err(ILError::catalog(
                "Transaction already committed or rolled back",
            ));
        }
        Ok(())
    }
}

#[async_trait::async_trait]
impl Transaction for SqliteTransaction {
    async fn query(&mut self, sql: &str, schema: CatalogSchemaRef) -> ILResult<RowStream> {
        trace!("sqlite txn query: {sql}");
        self.check_done()?;
        let mut stmt = self
            .conn
            .prepare(sql)
            .map_err(|e| ILError::catalog(format!("failed to prepare sqlite stmt: {sql} {e}")))?;
        let mut sqlite_rows = stmt
            .query([])
            .map_err(|e| ILError::catalog(format!("failed to query sqlite stmt: {sql} {e}")))?;

        let mut rows: Vec<Row> = Vec::new();
        while let Some(sqlite_row) = sqlite_rows
            .next()
            .map_err(|e| ILError::catalog(format!("failed to get next sqlite row: {e}")))?
        {
            let row = sqlite_row_to_row(sqlite_row, &schema)?;
            rows.push(row);
        }
        Ok(Box::pin(futures::stream::iter(rows).map(Ok)))
    }

    async fn execute(&mut self, sql: &str) -> ILResult<usize> {
        trace!("sqlite txn execute: {sql}");
        self.check_done()?;
        self.conn
            .execute(sql, [])
            .map_err(|e| ILError::catalog(format!("failed to execute sqlite stmt: {sql} {e}")))
    }

    async fn execute_batch(&mut self, sqls: &[String]) -> ILResult<()> {
        trace!("sqlite txn execute batch: {:?}", sqls);
        self.check_done()?;
        let sql = sqls.join(";");
        self.conn
            .execute_batch(&sql)
            .map_err(|e| ILError::catalog(format!("failed to execute sqlite batch: {sql} {e}")))
    }

    async fn commit(&mut self) -> ILResult<()> {
        trace!("sqlite txn commit");
        self.check_done()?;
        self.conn
            .execute_batch("COMMIT")
            .map_err(|e| ILError::catalog(format!("failed to commit sqlite txn: {e}")))?;
        self.done = true;
        Ok(())
    }

    async fn rollback(&mut self) -> ILResult<()> {
        trace!("sqlite txn rollback");
        self.check_done()?;
        self.conn
            .execute_batch("ROLLBACK")
            .map_err(|e| ILError::catalog(format!("failed to rollback sqlite txn: {e}")))?;
        self.done = true;
        Ok(())
    }
}

impl Drop for SqliteTransaction {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        if let Err(e) = self.conn.execute_batch("ROLLBACK") {
            error!("[indexlake] failed to rollback sqlite txn: {e}");
        }
    }
}

fn sqlite_row_to_row(sqlite_row: &rusqlite::Row, schema: &CatalogSchemaRef) -> ILResult<Row> {
    let mut row_values = Vec::new();
    let err_mapping =
        |e: rusqlite::Error| ILError::catalog(format!("failed to get row value: {e}"));
    for (idx, field) in schema.columns.iter().enumerate() {
        let scalar = match field.data_type {
            CatalogDataType::Boolean => {
                let v: Option<bool> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Boolean(v)
            }
            CatalogDataType::Int8 => {
                let v: Option<i8> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Int8(v)
            }
            CatalogDataType::Int16 => {
                let v: Option<i16> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Int16(v)
            }
            CatalogDataType::Int32 => {
                let v: Option<i32> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Int32(v)
            }
            CatalogDataType::Int64 => {
                let v: Option<i64> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Int64(v)
            }
            CatalogDataType::UInt8 => {
                let v: Option<u8> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::UInt8(v)
            }
            CatalogDataType::UInt16 => {
                let v: Option<u16> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::UInt16(v)
            }
            CatalogDataType::UInt32 => {
                let v: Option<u32> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::UInt32(v)
            }
            CatalogDataType::UInt64 => {
                let v: Option<f64> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::UInt64(v.map(|v| v as u64))
            }
            CatalogDataType::Float32 => {
                let v: Option<f32> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Float32(v)
            }
            CatalogDataType::Float64 => {
                let v: Option<f64> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Float64(v)
            }
            CatalogDataType::Utf8 => {
                let v: Option<String> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Utf8(v)
            }
            CatalogDataType::Binary => {
                let v: Option<Vec<u8>> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Binary(v)
            }
            CatalogDataType::Uuid => {
                let v: Option<Vec<u8>> = sqlite_row.get(idx).map_err(err_mapping)?;
                Scalar::Binary(v)
            }
        };
        if !field.nullable && scalar.is_null() {
            return Err(ILError::catalog(format!(
                "column {} is not nullable but got null value",
                field.name
            )));
        }
        row_values.push(scalar);
    }
    Ok(Row::new(schema.clone(), row_values))
}
