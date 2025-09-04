mod create;
mod delete;
mod insert;
mod query;
mod update;

use std::sync::Arc;

use futures::TryStreamExt;
use uuid::Uuid;

use crate::catalog::{Catalog, CatalogDatabase, CatalogSchemaRef, Row, Transaction};
use crate::{ILError, ILResult};

pub(crate) struct TransactionHelper {
    pub(crate) transaction: Box<dyn Transaction>,
    pub(crate) database: CatalogDatabase,
}

impl TransactionHelper {
    pub(crate) async fn new(catalog: &Arc<dyn Catalog>) -> ILResult<Self> {
        let transaction = catalog.transaction().await?;
        Ok(Self {
            transaction,
            database: catalog.database(),
        })
    }

    pub(crate) async fn query_rows(
        &mut self,
        sql: &str,
        schema: CatalogSchemaRef,
    ) -> ILResult<Vec<Row>> {
        let stream = self.transaction.query(sql, schema).await?;
        stream.try_collect::<Vec<_>>().await
    }

    pub(crate) async fn query_single(
        &mut self,
        sql: &str,
        schema: CatalogSchemaRef,
    ) -> ILResult<Option<Row>> {
        let stream = self.transaction.query(sql, schema).await?;
        let mut rows = stream.try_collect::<Vec<_>>().await?;
        if rows.len() > 1 {
            return Err(ILError::internal(format!(
                "Multiple rows found for sql {sql}"
            )));
        }
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows.remove(0)))
        }
    }

    pub(crate) async fn commit(&mut self) -> ILResult<()> {
        self.transaction.commit().await
    }

    #[allow(dead_code)]
    pub(crate) async fn rollback(&mut self) -> ILResult<()> {
        self.transaction.rollback().await
    }

    pub(crate) async fn truncate_inline_row_table(&mut self, table_id: &Uuid) -> ILResult<()> {
        match self.database {
            CatalogDatabase::Sqlite => {
                self.transaction
                    .execute_batch(&[format!("DELETE FROM {}", inline_row_table_name(table_id))])
                    .await
            }
            CatalogDatabase::Postgres => {
                self.transaction
                    .execute_batch(&[format!(
                        "TRUNCATE TABLE {}",
                        inline_row_table_name(table_id)
                    )])
                    .await
            }
        }
    }

    pub(crate) async fn drop_inline_row_table(&mut self, table_id: &Uuid) -> ILResult<()> {
        self.transaction
            .execute_batch(&[format!("DROP TABLE {}", inline_row_table_name(table_id))])
            .await?;
        Ok(())
    }
}

#[derive(Clone)]
pub(crate) struct CatalogHelper {
    pub(crate) catalog: Arc<dyn Catalog>,
}

impl CatalogHelper {
    pub(crate) fn new(catalog: Arc<dyn Catalog>) -> Self {
        Self { catalog }
    }

    pub(crate) async fn query_rows(
        &self,
        sql: &str,
        schema: CatalogSchemaRef,
    ) -> ILResult<Vec<Row>> {
        let stream = self.catalog.query(sql, schema).await?;
        stream.try_collect::<Vec<_>>().await
    }

    pub(crate) async fn query_single(
        &self,
        sql: &str,
        schema: CatalogSchemaRef,
    ) -> ILResult<Option<Row>> {
        let stream = self.catalog.query(sql, schema).await?;
        let mut rows = stream.try_collect::<Vec<_>>().await?;
        if rows.len() > 1 {
            return Err(ILError::internal(format!(
                "Multiple rows found for sql {sql}"
            )));
        }
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(rows.remove(0)))
        }
    }
}

pub(crate) fn inline_row_table_name(table_id: &Uuid) -> String {
    format!("indexlake_inline_{}", hex::encode(table_id.as_bytes()))
}
