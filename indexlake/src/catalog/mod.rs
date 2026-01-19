mod database;
mod helper;
mod record;
mod row;
mod scalar;
mod schema;

use arrow_schema::Schema;
pub use database::*;
pub(crate) use helper::*;
pub use record::*;
pub use row::*;
pub use scalar::*;
pub use schema::*;

use futures::Stream;
use uuid::Uuid;

use crate::ILResult;
use crate::expr::Expr;
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

    async fn truncate(&self, table_name: &str) -> ILResult<()>;

    async fn size(&self, table_name: &str) -> ILResult<usize>;

    fn sql_identifier(&self, ident: &str) -> String;

    fn sql_binary_literal(&self, value: &[u8]) -> String;

    fn sql_uuid_literal(&self, value: &Uuid) -> String;

    fn sql_string_literal(&self, value: &str) -> String;

    fn supports_filter(&self, filter: &Expr, schema: &Schema) -> ILResult<bool>;

    fn unparse_expr(&self, expr: &Expr) -> ILResult<String>;

    fn unparse_catalog_data_type(&self, data_type: CatalogDataType) -> String;
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
