use arrow::datatypes::Schema;
use futures::StreamExt;
use indexlake::catalog::{
    Catalog, CatalogDataType, CatalogDatabase, CatalogSchemaRef, Row, RowStream, Scalar,
    Transaction,
};
use indexlake::expr::{BinaryExpr, Expr};
use indexlake::{ILError, ILResult};
use log::{error, trace};
use rusqlite::OpenFlags;
use std::path::PathBuf;
use uuid::Uuid;

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

    async fn truncate(&self, table_name: &str) -> ILResult<()> {
        let conn = rusqlite::Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| ILError::catalog(format!("failed to open sqlite db: {e}")))?;
        conn.execute(
            &format!("DELETE FROM {}", self.sql_identifier(table_name)),
            [],
        )
        .map_err(|e| {
            ILError::catalog(format!(
                "failed to truncate table {table_name} on sqlite: {e}"
            ))
        })?;
        Ok(())
    }

    async fn size(&self, table_name: &str) -> ILResult<usize> {
        let conn = rusqlite::Connection::open_with_flags(
            &self.path,
            OpenFlags::SQLITE_OPEN_READ_WRITE
                | OpenFlags::SQLITE_OPEN_NO_MUTEX
                | OpenFlags::SQLITE_OPEN_URI,
        )
        .map_err(|e| ILError::catalog(format!("failed to open sqlite db: {e}")))?;
        let size: usize = conn
            .query_row(
                &format!("SELECT SUM(pgsize) FROM dbstat WHERE name='{table_name}'"),
                [],
                |row| row.get(0),
            )
            .map_err(|e| {
                ILError::catalog(format!(
                    "failed to get size of table {table_name} on sqlite: {e}"
                ))
            })?;
        Ok(size)
    }

    fn sql_identifier(&self, ident: &str) -> String {
        format!("`{ident}`")
    }

    fn sql_binary_literal(&self, value: &[u8]) -> String {
        format!("X'{}'", hex::encode(value))
    }

    fn sql_uuid_literal(&self, value: &Uuid) -> String {
        self.sql_binary_literal(value.as_bytes())
    }

    fn sql_string_literal(&self, value: &str) -> String {
        let value = value.replace("'", "''");
        format!("'{value}'")
    }

    // TODO impl this
    fn supports_filter(&self, filter: &Expr, _schema: &Schema) -> ILResult<bool> {
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

    fn unparse_expr(&self, expr: &Expr) -> ILResult<String> {
        match expr {
            Expr::Column(name) => Ok(self.sql_identifier(name)),
            Expr::Literal(literal) => literal.value.to_sql(self),
            Expr::BinaryExpr(binary_expr) => {
                let left = self.unparse_expr(&binary_expr.left)?;
                let right = self.unparse_expr(&binary_expr.right)?;
                Ok(format!("({} {} {})", left, binary_expr.op, right))
            }
            Expr::Not(expr) => Ok(format!("NOT {}", self.unparse_expr(expr)?)),
            Expr::IsNull(expr) => Ok(format!("{} IS NULL", self.unparse_expr(expr)?)),
            Expr::IsNotNull(expr) => Ok(format!("{} IS NOT NULL", self.unparse_expr(expr)?)),
            Expr::InList(in_list) => {
                let list = in_list
                    .list
                    .iter()
                    .map(|expr| self.unparse_expr(expr))
                    .collect::<ILResult<Vec<_>>>()?
                    .join(", ");
                Ok(format!(
                    "{} IN ({})",
                    self.unparse_expr(&in_list.expr)?,
                    list
                ))
            }
            Expr::Function(_) => Err(ILError::invalid_input(
                "Function can only be used for index",
            )),
            Expr::Like(like) => {
                let expr = self.unparse_expr(&like.expr)?;
                let pattern = self.unparse_expr(&like.pattern)?;
                // For case-sensitive LIKE, SQLite requires `PRAGMA case_sensitive_like = ON;`
                // to be set on the connection. This function only generates the SQL string
                // and does not set the PRAGMA.
                // For case-insensitive ILIKE, we use the `UPPER()` function on both
                // the expression and the pattern to ensure case-insensitivity.
                match (like.negated, like.case_insensitive) {
                    (false, false) => Ok(format!("{expr} LIKE {pattern}")),
                    (true, false) => Ok(format!("{expr} NOT LIKE {pattern}")),
                    (false, true) => Ok(format!("UPPER({expr}) LIKE UPPER({pattern})")),
                    (true, true) => Ok(format!("UPPER({expr}) NOT LIKE UPPER({pattern})")),
                }
            }
            Expr::Cast(cast) => {
                let catalog_datatype = CatalogDataType::from_arrow(&cast.cast_type)?;
                let expr_sql = self.unparse_expr(&cast.expr)?;
                Ok(format!(
                    "CAST({} AS {})",
                    expr_sql,
                    self.unparse_catalog_data_type(catalog_datatype),
                ))
            }
            Expr::TryCast(_) => Err(ILError::invalid_input("TRY_CAST is not supported in SQL")),
            Expr::Negative(expr) => Ok(format!("-{}", self.unparse_expr(expr)?)),
            Expr::Case(case) => {
                let mut sql = String::new();
                sql.push_str("CASE");
                for (when, then) in &case.when_then {
                    sql.push_str(&format!(
                        " WHEN {} THEN {}",
                        self.unparse_expr(when)?,
                        self.unparse_expr(then)?,
                    ));
                }
                if let Some(else_expr) = &case.else_expr {
                    sql.push_str(&format!(" ELSE {}", self.unparse_expr(else_expr)?));
                }
                sql.push_str(" END");
                Ok(sql)
            }
        }
    }

    fn unparse_catalog_data_type(&self, data_type: CatalogDataType) -> String {
        match data_type {
            CatalogDataType::Boolean => "BOOLEAN".to_string(),
            CatalogDataType::Int8 => "TINYINT".to_string(),
            CatalogDataType::Int16 => "SMALLINT".to_string(),
            CatalogDataType::Int32 => "INTEGER".to_string(),
            CatalogDataType::Int64 => "BIGINT".to_string(),
            CatalogDataType::UInt8 => "TINYINT UNSIGNED".to_string(),
            CatalogDataType::UInt16 => "SMALLINT UNSIGNED".to_string(),
            CatalogDataType::UInt32 => "INTEGER UNSIGNED".to_string(),
            CatalogDataType::UInt64 => "BIGINT UNSIGNED".to_string(),
            CatalogDataType::Float32 => "FLOAT".to_string(),
            CatalogDataType::Float64 => "DOUBLE".to_string(),
            CatalogDataType::Utf8 => "VARCHAR".to_string(),
            CatalogDataType::Binary => "BLOB".to_string(),
            CatalogDataType::Uuid => "BLOB".to_string(),
        }
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
