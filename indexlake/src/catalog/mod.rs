mod database;
mod helper;
mod record;
mod row;
mod scalar;
mod schema;

pub use database::*;
pub(crate) use helper::*;
pub(crate) use record::*;
pub use row::*;
pub use scalar::*;
pub use schema::*;

use futures::Stream;

use crate::ILResult;
use arrow::datatypes::{DataType, Field, FieldRef};
use std::fmt::Debug;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

pub type RowStream<'a> = Pin<Box<dyn Stream<Item = ILResult<Row>> + Send + 'a>>;

pub static INTERNAL_ROW_ID_FIELD_NAME: &str = "_indexlake_row_id";
pub static INTERNAL_ROW_ID_FIELD_REF: LazyLock<FieldRef> = LazyLock::new(|| {
    Arc::new(Field::new(
        INTERNAL_ROW_ID_FIELD_NAME,
        DataType::FixedSizeBinary(16),
        false,
    ))
});

#[async_trait::async_trait]
pub trait Catalog: Debug + Send + Sync {
    fn database(&self) -> CatalogDatabase;

    async fn query(&self, sql: &str, schema: CatalogSchemaRef) -> ILResult<RowStream<'static>>;

    /// Begin a new transaction.
    async fn transaction(&self) -> ILResult<Box<dyn Transaction>>;
}

// Transaction should be rolled back when dropped.
#[async_trait::async_trait]
pub trait Transaction: Debug + Send {
    /// Execute a query and return a stream of rows.
    async fn query<'a>(
        &'a mut self,
        sql: &str,
        schema: CatalogSchemaRef,
    ) -> ILResult<RowStream<'a>>;

    /// Execute a SQL statement.
    async fn execute(&mut self, sql: &str) -> ILResult<usize>;

    /// Execute a batch of SQL statements.
    async fn execute_batch(&mut self, sqls: &[String]) -> ILResult<()>;

    /// Commit the transaction.
    async fn commit(&mut self) -> ILResult<()>;

    /// Rollback the transaction.
    async fn rollback(&mut self) -> ILResult<()>;
}
