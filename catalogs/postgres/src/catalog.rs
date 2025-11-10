use arrow::datatypes::Schema;
use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use bb8_postgres::tokio_postgres::NoTls;
use futures::StreamExt;
use indexlake::catalog::{
    Catalog, CatalogDataType, CatalogSchemaRef, Row, RowStream, Scalar, Transaction,
};
use indexlake::expr::{BinaryExpr, Expr};
use indexlake::{ILError, ILResult};
use log::{error, trace};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct PostgresCatalog {
    pool: Pool<PostgresConnectionManager<NoTls>>,
}

impl PostgresCatalog {
    pub fn new(pool: Pool<PostgresConnectionManager<NoTls>>) -> Self {
        Self { pool }
    }
}

#[async_trait::async_trait]
impl Catalog for PostgresCatalog {
    async fn query(&self, sql: &str, schema: CatalogSchemaRef) -> ILResult<RowStream<'static>> {
        trace!("postgres query: {sql}");
        let conn = self
            .pool
            .get_owned()
            .await
            .map_err(|e| ILError::catalog(format!("failed to get postgres connection: {e}")))?;
        let pg_row_stream = conn
            .query_raw(sql, Vec::<String>::new())
            .await
            .map_err(|e| ILError::catalog(format!("failed to query postgres: {sql} {e}")))?;

        let stream = pg_row_stream.map(move |row| {
            let pg_row =
                row.map_err(|e| ILError::catalog(format!("failed to get postgres row: {e}")))?;
            pg_row_to_row(&pg_row, &schema)
        });
        Ok(Box::pin(stream))
    }

    async fn transaction(&self) -> ILResult<Box<dyn Transaction>> {
        let conn = self
            .pool
            .get_owned()
            .await
            .map_err(|e| ILError::catalog(format!("failed to get postgres connection: {e}")))?;
        conn.batch_execute("START TRANSACTION")
            .await
            .map_err(|e| ILError::catalog(format!("failed to start postgres txn: {e}")))?;
        Ok(Box::new(PostgresTransaction { conn, done: false }))
    }

    async fn truncate(&self, table_name: &str) -> ILResult<()> {
        let conn = self
            .pool
            .get()
            .await
            .map_err(|e| ILError::catalog(format!("failed to get postgres connection: {e}")))?;
        conn.execute(
            &format!("TRUNCATE TABLE {}", self.sql_identifier(table_name)),
            &[],
        )
        .await
        .map(|r| r as usize)
        .map_err(|e| {
            ILError::catalog(format!(
                "failed to truncate table {table_name} on postgres: {e}"
            ))
        })?;
        Ok(())
    }

    async fn size(&self, table_name: &str) -> ILResult<usize> {
        let conn = self
            .pool
            .get()
            .await
            .map_err(|e| ILError::catalog(format!("failed to get postgres connection: {e}")))?;
        let row = conn
            .query_one(
                &format!("SELECT pg_total_relation_size('{table_name}')"),
                &[],
            )
            .await
            .map_err(|e| {
                ILError::catalog(format!(
                    "failed to get size of table {table_name} on postgres: {e}"
                ))
            })?;
        let size = row.get::<_, i64>(0);
        Ok(size as usize)
    }

    fn sql_identifier(&self, ident: &str) -> String {
        format!("\"{ident}\"")
    }

    fn sql_binary_literal(&self, value: &[u8]) -> String {
        format!("E'\\\\x{}'", hex::encode(value))
    }

    fn sql_uuid_literal(&self, value: &Uuid) -> String {
        format!("'{value}'")
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
                match (like.negated, like.case_insensitive) {
                    (true, true) => Ok(format!("{expr} NOT ILIKE {pattern}")),
                    (true, false) => Ok(format!("{expr} NOT LIKE {pattern}")),
                    (false, true) => Ok(format!("{expr} ILIKE {pattern}")),
                    (false, false) => Ok(format!("{expr} LIKE {pattern}")),
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
            CatalogDataType::Int8 => "SMALLINT".to_string(),
            CatalogDataType::Int16 => "SMALLINT".to_string(),
            CatalogDataType::Int32 => "INTEGER".to_string(),
            CatalogDataType::Int64 => "BIGINT".to_string(),
            CatalogDataType::UInt8 => "SMALLINT".to_string(),
            CatalogDataType::UInt16 => "INTEGER".to_string(),
            CatalogDataType::UInt32 => "BIGINT".to_string(),
            CatalogDataType::UInt64 => "FLOAT4".to_string(),
            CatalogDataType::Float32 => "FLOAT4".to_string(),
            CatalogDataType::Float64 => "FLOAT8".to_string(),
            CatalogDataType::Utf8 => "VARCHAR".to_string(),
            CatalogDataType::Binary => "BYTEA".to_string(),
            CatalogDataType::Uuid => "UUID".to_string(),
        }
    }
}

#[derive(Debug)]
pub struct PostgresTransaction {
    conn: bb8::PooledConnection<'static, PostgresConnectionManager<NoTls>>,
    done: bool,
}

impl PostgresTransaction {
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
impl Transaction for PostgresTransaction {
    async fn query(&mut self, sql: &str, schema: CatalogSchemaRef) -> ILResult<RowStream> {
        trace!("postgres txn query: {sql}");
        self.check_done()?;

        let pg_row_stream = self
            .conn
            .query_raw(sql, Vec::<String>::new())
            .await
            .map_err(|e| ILError::catalog(format!("failed to query postgres: {sql} {e}")))?;

        let stream = pg_row_stream.map(move |row| {
            let pg_row =
                row.map_err(|e| ILError::catalog(format!("failed to get postgres row: {e}")))?;
            pg_row_to_row(&pg_row, &schema)
        });
        Ok(Box::pin(stream))
    }

    async fn execute(&mut self, sql: &str) -> ILResult<usize> {
        trace!("postgres txn execute: {sql}");
        self.check_done()?;
        self.conn
            .execute(sql, &[])
            .await
            .map(|r| r as usize)
            .map_err(|e| ILError::catalog(format!("failed to execute postgres: {sql} {e}")))
    }

    async fn execute_batch(&mut self, sqls: &[String]) -> ILResult<()> {
        trace!("postgres txn execute batch: {:?}", sqls);
        self.check_done()?;
        let sql = sqls.join(";");
        self.conn
            .batch_execute(&sql)
            .await
            .map_err(|e| ILError::catalog(format!("failed to execute batch postgres: {sql} {e}")))
    }

    async fn commit(&mut self) -> ILResult<()> {
        trace!("postgres txn commit");
        self.check_done()?;
        self.conn
            .batch_execute("COMMIT")
            .await
            .map_err(|e| ILError::catalog(format!("failed to commit postgres txn: {e}")))?;
        self.done = true;
        Ok(())
    }

    async fn rollback(&mut self) -> ILResult<()> {
        trace!("postgres txn rollback");
        self.check_done()?;
        self.conn
            .batch_execute("ROLLBACK")
            .await
            .map_err(|e| ILError::catalog(format!("failed to rollback postgres txn: {e}")))?;
        self.done = true;
        Ok(())
    }
}

impl Drop for PostgresTransaction {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                if let Err(e) = self.conn.batch_execute("ROLLBACK").await {
                    error!("[indexlake] failed to rollback postgres txn: {e}");
                }
            });
        });
    }
}

fn pg_row_to_row(
    pg_row: &bb8_postgres::tokio_postgres::Row,
    schema: &CatalogSchemaRef,
) -> ILResult<Row> {
    let mut values = Vec::new();
    let err_mapping = |e: bb8_postgres::tokio_postgres::error::Error| {
        ILError::catalog(format!("failed to get row value: {e}"))
    };
    for (idx, field) in schema.columns.iter().enumerate() {
        let scalar = match field.data_type {
            CatalogDataType::Boolean => {
                let v: Option<bool> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Boolean(v)
            }
            CatalogDataType::Int8 => {
                let v: Option<i16> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Int8(v.map(|v| v as i8))
            }
            CatalogDataType::Int16 => {
                let v: Option<i16> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Int16(v)
            }
            CatalogDataType::Int32 => {
                let v: Option<i32> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Int32(v)
            }
            CatalogDataType::Int64 => {
                let v: Option<i64> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Int64(v)
            }
            CatalogDataType::UInt8 => {
                let v: Option<i16> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::UInt8(v.map(|v| v as u8))
            }
            CatalogDataType::UInt16 => {
                let v: Option<i32> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::UInt16(v.map(|v| v as u16))
            }
            CatalogDataType::UInt32 => {
                let v: Option<i64> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::UInt32(v.map(|v| v as u32))
            }
            CatalogDataType::UInt64 => {
                let v: Option<f32> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::UInt64(v.map(|v| v as u64))
            }
            CatalogDataType::Float32 => {
                let v: Option<f32> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Float32(v)
            }
            CatalogDataType::Float64 => {
                let v: Option<f64> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Float64(v)
            }
            CatalogDataType::Utf8 => {
                let v: Option<String> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Utf8(v)
            }
            CatalogDataType::Binary => {
                let v: Option<Vec<u8>> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Binary(v)
            }
            CatalogDataType::Uuid => {
                let v: Option<uuid::Uuid> = pg_row.try_get(idx).map_err(err_mapping)?;
                Scalar::Binary(v.map(|v| v.as_bytes().to_vec()))
            }
        };
        if !field.nullable && scalar.is_null() {
            return Err(ILError::catalog(format!(
                "column {} is not nullable but got null value",
                field.name
            )));
        }
        values.push(scalar);
    }
    Ok(Row::new(schema.clone(), values))
}
