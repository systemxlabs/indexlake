use bb8::Pool;
use bb8_postgres::{PostgresConnectionManager, tokio_postgres::NoTls};
use futures::StreamExt;
use indexlake::{
    ILError, ILResult,
    catalog::{Catalog, CatalogDatabase, RowStream, Transaction},
    record::{DataType, Row, Scalar, SchemaRef},
};
use log::debug;

#[derive(Debug, Clone)]
pub struct PostgresCatalog {
    pool: Pool<PostgresConnectionManager<NoTls>>,
}

impl PostgresCatalog {
    pub async fn try_new(
        host: &str,
        port: u16,
        user: &str,
        password: &str,
        dbname: Option<&str>,
    ) -> ILResult<Self> {
        let mut config = bb8_postgres::tokio_postgres::config::Config::new();
        config.host(host).port(port).user(user).password(password);
        if let Some(dbname) = dbname {
            config.dbname(dbname);
        }
        let manager = PostgresConnectionManager::new(config, NoTls);
        let pool = Pool::builder()
            .build(manager)
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        Ok(Self { pool })
    }
}

#[async_trait::async_trait]
impl Catalog for PostgresCatalog {
    fn database(&self) -> CatalogDatabase {
        CatalogDatabase::Postgres
    }

    async fn transaction(&self) -> ILResult<Box<dyn Transaction>> {
        let conn = self
            .pool
            .get_owned()
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        conn.batch_execute("START TRANSACTION")
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        Ok(Box::new(PostgresTransaction { conn, done: false }))
    }
}

#[derive(Debug)]
pub struct PostgresTransaction {
    conn: bb8::PooledConnection<'static, PostgresConnectionManager<NoTls>>,
    done: bool,
}

#[async_trait::async_trait]
impl Transaction for PostgresTransaction {
    async fn query(&mut self, sql: &str, schema: SchemaRef) -> ILResult<RowStream> {
        debug!("postgres txn query: {sql}");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }

        let pg_row_stream = self
            .conn
            .query_raw(sql, Vec::<String>::new())
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;

        let stream = pg_row_stream.map(move |row| {
            let pg_row = row.map_err(|e| ILError::CatalogError(e.to_string()))?;
            pg_row_to_row(&pg_row, &schema)
        });
        Ok(Box::pin(stream))
    }

    async fn execute(&mut self, sql: &str) -> ILResult<usize> {
        debug!("postgres txn execute: {sql}");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .execute(sql, &[])
            .await
            .map(|r| r as usize)
            .map_err(|e| ILError::CatalogError(e.to_string()))
    }

    async fn execute_batch(&mut self, sqls: &[String]) -> ILResult<()> {
        debug!("postgres txn execute batch: {:?}", sqls);
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .batch_execute(sqls.join(";").as_str())
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))
    }

    async fn commit(&mut self) -> ILResult<()> {
        debug!("postgres txn commit");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .batch_execute("COMMIT")
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
        self.done = true;
        Ok(())
    }

    async fn rollback(&mut self) -> ILResult<()> {
        debug!("postgres txn rollback");
        if self.done {
            return Err(ILError::CatalogError(
                "Transaction already committed or rolled back".to_string(),
            ));
        }
        self.conn
            .batch_execute("ROLLBACK")
            .await
            .map_err(|e| ILError::CatalogError(e.to_string()))?;
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
                self.conn
                    .batch_execute("ROLLBACK")
                    .await
                    .expect("rollback failed");
            });
        });
    }
}

fn pg_row_to_row(pg_row: &bb8_postgres::tokio_postgres::Row, schema: &SchemaRef) -> ILResult<Row> {
    let mut values = Vec::new();
    for (idx, field) in schema.fields.iter().enumerate() {
        let scalar = match field.data_type {
            DataType::Int16 => {
                let v: Option<i16> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int16(v)
            }
            DataType::Int32 => {
                let v: Option<i32> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int32(v)
            }
            DataType::Int64 => {
                let v: Option<i64> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Int64(v)
            }
            DataType::Float32 => {
                let v: Option<f32> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Float32(v)
            }
            DataType::Float64 => {
                let v: Option<f64> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Float64(v)
            }
            DataType::Utf8 => {
                let v: Option<String> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Utf8(v)
            }
            DataType::Binary => {
                let v: Option<Vec<u8>> = pg_row
                    .try_get(idx)
                    .map_err(|e| ILError::CatalogError(e.to_string()))?;
                Scalar::Binary(v)
            }
            DataType::Boolean => {
                let v: Option<bool> = pg_row
                    .try_get(idx)
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
        values.push(scalar);
    }
    Ok(Row::new(schema.clone(), values))
}
