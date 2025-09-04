use std::sync::Arc;

use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{
    CatalogDataType, CatalogHelper, CatalogSchema, CatalogSchemaRef, Column, DataFileRecord,
    FieldRecord, INTERNAL_ROW_ID_FIELD_NAME, IndexFileRecord, IndexRecord, InlineIndexRecord, Row,
    RowStream, TableRecord, TransactionHelper, inline_row_table_name,
};
use crate::expr::Expr;

impl TransactionHelper {
    pub(crate) async fn get_namespace_id(
        &mut self,
        namespace_name: &str,
    ) -> ILResult<Option<Uuid>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "namespace_id",
            CatalogDataType::Uuid,
            false,
        )]));
        let row = self
            .query_single(
                &format!(
                    "SELECT namespace_id FROM indexlake_namespace WHERE namespace_name = '{namespace_name}'"
                ),
                schema,
            )
            .await?;
        match row {
            Some(row) => row.uuid(0),
            None => Ok(None),
        }
    }

    pub(crate) async fn get_table_id(
        &mut self,
        namespace_id: &Uuid,
        table_name: &str,
    ) -> ILResult<Option<Uuid>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "table_id",
            CatalogDataType::Uuid,
            false,
        )]));
        let row = self.query_single(&format!("SELECT table_id FROM indexlake_table WHERE namespace_id = {} AND table_name = '{table_name}'", self.database.sql_uuid_literal(namespace_id)), schema).await?;
        match row {
            Some(row) => row.uuid(0),
            None => Ok(None),
        }
    }

    pub(crate) async fn scan_inline_rows(
        &mut self,
        table_id: &Uuid,
        schema: &CatalogSchemaRef,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> ILResult<RowStream<'_>> {
        let filter_strs = filters
            .iter()
            .map(|f| f.to_sql(self.database))
            .collect::<Result<Vec<_>, _>>()?;

        let where_clause = if filter_strs.is_empty() {
            "".to_string()
        } else {
            format!(" WHERE {}", filter_strs.join(" AND "))
        };

        let limit_clause = limit
            .map(|limit| format!(" LIMIT {limit}"))
            .unwrap_or_default();
        self.transaction
            .query(
                &format!(
                    "SELECT {}  FROM {}{where_clause}{limit_clause}",
                    schema.select_items(self.database).join(", "),
                    inline_row_table_name(table_id),
                ),
                Arc::clone(schema),
            )
            .await
    }

    pub(crate) async fn count_inline_rows(&mut self, table_id: &Uuid) -> ILResult<i64> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "count",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self
            .query_rows(
                &format!("SELECT COUNT(1) FROM {}", inline_row_table_name(table_id)),
                schema,
            )
            .await?;
        let count = rows[0].int64(0)?.expect("count is not null");
        Ok(count)
    }

    pub(crate) async fn get_data_files(
        &mut self,
        table_id: &Uuid,
    ) -> ILResult<Vec<DataFileRecord>> {
        let schema = Arc::new(DataFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_data_file WHERE table_id = {}",
                    schema.select_items(self.database).join(", "),
                    self.database.sql_uuid_literal(table_id),
                ),
                schema,
            )
            .await?;
        let mut data_files = Vec::with_capacity(rows.len());
        for row in rows {
            data_files.push(DataFileRecord::from_row(row)?);
        }
        Ok(data_files)
    }

    pub(crate) async fn get_index_id(
        &mut self,
        table_id: &Uuid,
        index_name: &str,
    ) -> ILResult<Option<Uuid>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "index_id",
            CatalogDataType::Uuid,
            false,
        )]));
        let row = self.query_single(&format!("SELECT index_id FROM indexlake_index WHERE table_id = {} AND index_name = '{index_name}'", self.database.sql_uuid_literal(table_id)), schema).await?;
        match row {
            Some(row) => row.uuid(0),
            None => Ok(None),
        }
    }
}

impl CatalogHelper {
    pub(crate) async fn get_namespace_id(&self, namespace_name: &str) -> ILResult<Option<Uuid>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "namespace_id",
            CatalogDataType::Uuid,
            false,
        )]));
        let row = self
            .query_single(
                &format!(
                    "SELECT namespace_id FROM indexlake_namespace WHERE namespace_name = '{namespace_name}'"
                ),
                schema,
            )
            .await?;
        match row {
            Some(row) => row.uuid(0),
            None => Ok(None),
        }
    }

    pub(crate) async fn get_table(
        &self,
        namespace_id: &Uuid,
        table_name: &str,
    ) -> ILResult<Option<TableRecord>> {
        let schema = Arc::new(TableRecord::catalog_schema());
        let row = self
            .query_single(
                &format!(
                    "SELECT {} FROM indexlake_table WHERE namespace_id = {} AND table_name = '{table_name}'",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(namespace_id)
                ),
                schema,
            )
            .await?;
        match row {
            Some(row) => Ok(Some(TableRecord::from_row(row)?)),
            None => Ok(None),
        }
    }

    pub(crate) async fn get_table_fields(&self, table_id: &Uuid) -> ILResult<Vec<FieldRecord>> {
        let catalog_schema = Arc::new(FieldRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_field WHERE table_id = {} order by field_id asc",
                    catalog_schema
                        .select_items(self.catalog.database())
                        .join(", "),
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                catalog_schema,
            )
            .await?;
        let mut field_records = Vec::with_capacity(rows.len());
        for row in rows {
            field_records.push(FieldRecord::from_row(row)?);
        }
        Ok(field_records)
    }

    pub(crate) async fn get_table_indexes(&self, table_id: &Uuid) -> ILResult<Vec<IndexRecord>> {
        let catalog_schema = Arc::new(IndexRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_index WHERE table_id = {}",
                    catalog_schema
                        .select_items(self.catalog.database())
                        .join(", "),
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                catalog_schema,
            )
            .await?;
        let mut indexes = Vec::with_capacity(rows.len());
        for row in rows {
            indexes.push(IndexRecord::from_row(row)?);
        }
        Ok(indexes)
    }

    pub(crate) async fn count_inline_rows(&self, table_id: &Uuid) -> ILResult<i64> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "count",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self
            .query_rows(
                &format!("SELECT COUNT(1) FROM {}", inline_row_table_name(table_id)),
                schema,
            )
            .await?;
        let count = rows[0].int64(0)?.expect("count is not null");
        Ok(count)
    }

    pub(crate) async fn scan_inline_rows(
        &self,
        table_id: &Uuid,
        table_schema: &CatalogSchemaRef,
        row_ids: Option<&[Uuid]>,
        filters: &[Expr],
    ) -> ILResult<RowStream<'static>> {
        if let Some(row_ids) = row_ids
            && row_ids.is_empty()
        {
            return Ok(Box::pin(futures::stream::empty::<ILResult<Row>>()));
        }

        let mut filter_strs = filters
            .iter()
            .map(|f| f.to_sql(self.catalog.database()))
            .collect::<Result<Vec<_>, _>>()?;

        if let Some(row_ids) = row_ids {
            filter_strs.push(format!(
                "{INTERNAL_ROW_ID_FIELD_NAME} IN ({})",
                row_ids
                    .iter()
                    .map(|id| self.catalog.database().sql_uuid_literal(id))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        let where_clause = if filter_strs.is_empty() {
            "".to_string()
        } else {
            format!(" WHERE {}", filter_strs.join(" AND "))
        };

        self.catalog
            .query(
                &format!(
                    "SELECT {} FROM {}{where_clause}",
                    table_schema
                        .select_items(self.catalog.database())
                        .join(", "),
                    inline_row_table_name(table_id),
                ),
                Arc::clone(table_schema),
            )
            .await
    }

    pub(crate) async fn count_data_files(&self, table_id: &Uuid) -> ILResult<i64> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "count",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self
            .query_rows(
                &format!(
                    "SELECT COUNT(1) FROM indexlake_data_file WHERE table_id = {}",
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                schema,
            )
            .await?;
        let count = rows[0].int64(0)?.expect("count is not null");
        Ok(count)
    }

    pub(crate) async fn get_partitioned_data_files(
        &self,
        table_id: &Uuid,
        offset: usize,
        limit: usize,
    ) -> ILResult<Vec<DataFileRecord>> {
        let schema = Arc::new(DataFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_data_file WHERE table_id = {} ORDER BY data_file_id ASC LIMIT {limit} OFFSET {offset}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(table_id),
                ),
                schema,
            )
            .await?;
        let mut data_files = Vec::with_capacity(rows.len());
        for row in rows {
            data_files.push(DataFileRecord::from_row(row)?);
        }
        Ok(data_files)
    }

    pub(crate) async fn get_data_files(&self, table_id: &Uuid) -> ILResult<Vec<DataFileRecord>> {
        let schema = Arc::new(DataFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_data_file WHERE table_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                schema,
            )
            .await?;
        let mut data_files = Vec::with_capacity(rows.len());
        for row in rows {
            data_files.push(DataFileRecord::from_row(row)?);
        }
        Ok(data_files)
    }

    pub(crate) async fn get_table_index_files(
        &self,
        table_id: &Uuid,
    ) -> ILResult<Vec<IndexFileRecord>> {
        let schema = Arc::new(IndexFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_index_file WHERE table_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(row)?);
        }
        Ok(index_files)
    }

    pub(crate) async fn get_index_files_by_index_id(
        &self,
        index_id: &Uuid,
    ) -> ILResult<Vec<IndexFileRecord>> {
        let schema = Arc::new(IndexFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_index_file WHERE index_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(index_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(row)?);
        }
        Ok(index_files)
    }

    pub(crate) async fn get_index_files_by_data_file_id(
        &self,
        data_file_id: &Uuid,
    ) -> ILResult<Vec<IndexFileRecord>> {
        let schema = Arc::new(IndexFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_index_file WHERE data_file_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(data_file_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(row)?);
        }
        Ok(index_files)
    }

    pub(crate) async fn get_index_file_by_index_id_and_data_file_id(
        &self,
        index_id: &Uuid,
        data_file_id: &Uuid,
    ) -> ILResult<Option<IndexFileRecord>> {
        let schema = Arc::new(IndexFileRecord::catalog_schema());
        let row = self
            .query_single(
                &format!(
                    "SELECT {} FROM indexlake_index_file WHERE index_id = {} AND data_file_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_literal(index_id),
                    self.catalog.database().sql_uuid_literal(data_file_id)
                ),
                schema,
            )
            .await?;
        match row {
            Some(row) => Ok(Some(IndexFileRecord::from_row(row)?)),
            None => Ok(None),
        }
    }

    pub(crate) async fn dump_task_exists(&self, table_id: &Uuid) -> ILResult<bool> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "table_id",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self
            .query_rows(
                &format!(
                    "SELECT table_id FROM indexlake_dump_task WHERE table_id = {}",
                    self.catalog.database().sql_uuid_literal(table_id)
                ),
                schema,
            )
            .await?;
        Ok(!rows.is_empty())
    }

    pub(crate) async fn get_inline_indexes(
        &self,
        index_ids: &[Uuid],
    ) -> ILResult<Vec<InlineIndexRecord>> {
        if index_ids.is_empty() {
            return Ok(Vec::new());
        }
        let schema = Arc::new(InlineIndexRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_inline_index WHERE index_id IN ({})",
                    schema.select_items(self.catalog.database()).join(", "),
                    index_ids
                        .iter()
                        .map(|id| self.catalog.database().sql_uuid_literal(id))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                schema,
            )
            .await?;
        let mut inline_indexes = Vec::with_capacity(rows.len());
        for row in rows {
            inline_indexes.push(InlineIndexRecord::from_row(row)?);
        }
        Ok(inline_indexes)
    }
}
