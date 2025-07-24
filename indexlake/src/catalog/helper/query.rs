use std::collections::BTreeMap;
use std::{collections::HashMap, sync::Arc};

use arrow::datatypes::{DataType, Field, FieldRef};
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, DataFileRecord, FieldRecord, IndexFileRecord, IndexRecord, RowsValidity,
    inline_row_table_name,
};
use crate::expr::Expr;
use crate::{
    ILError, ILResult,
    catalog::{
        CatalogDataType, CatalogSchema, CatalogSchemaRef, Column, INTERNAL_FLAG_FIELD_NAME,
        INTERNAL_ROW_ID_FIELD_NAME, Row,
    },
    catalog::{RowStream, TableRecord, TransactionHelper},
    table::TableConfig,
};

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
        let rows = self
            .query_rows(
                &format!(
                    "SELECT namespace_id FROM indexlake_namespace WHERE namespace_name = '{namespace_name}'"
                ),
                schema,
            )
            .await?;
        if rows.is_empty() {
            Ok(None)
        } else {
            let namespace_id_opt = rows[0].uuid(0)?;
            Ok(namespace_id_opt)
        }
    }

    pub(crate) async fn table_name_exists(
        &mut self,
        namespace_id: &Uuid,
        table_name: &str,
    ) -> ILResult<bool> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "table_id",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self.query_rows(&format!("SELECT table_id FROM indexlake_table WHERE namespace_id = {} AND table_name = '{table_name}'", self.database.sql_uuid_value(namespace_id)), schema).await?;
        Ok(rows.len() > 0)
    }

    pub(crate) async fn scan_inline_rows(
        &mut self,
        table_id: &Uuid,
        schema: &CatalogSchemaRef,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> ILResult<RowStream> {
        let mut filter_strs = filters
            .iter()
            .map(|f| f.to_sql(self.database))
            .collect::<Result<Vec<_>, _>>()?;
        filter_strs.push(format!("{INTERNAL_FLAG_FIELD_NAME} IS NULL"));

        let limit_clause = limit
            .map(|limit| format!(" LIMIT {limit}"))
            .unwrap_or_default();
        self.transaction
            .query(
                &format!(
                    "SELECT {}  FROM {} WHERE {}{limit_clause}",
                    schema.select_items(self.database).join(", "),
                    inline_row_table_name(table_id),
                    filter_strs.join(" AND ")
                ),
                Arc::clone(schema),
            )
            .await
    }

    pub(crate) async fn scan_inline_row_ids_by_flag(
        &mut self,
        table_id: &Uuid,
        flag: &str,
    ) -> ILResult<Vec<i64>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "row_id",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self.query_rows(
            &format!("SELECT {INTERNAL_ROW_ID_FIELD_NAME} FROM {} WHERE {INTERNAL_FLAG_FIELD_NAME} = '{flag}'", inline_row_table_name(table_id)),
            schema,
        ).await?;
        let mut row_ids = Vec::with_capacity(rows.len());
        for row in rows {
            row_ids.push(row.int64(0)?.expect("row_id is not null"));
        }
        Ok(row_ids)
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
                    self.database.sql_uuid_value(table_id),
                ),
                schema,
            )
            .await?;
        let mut data_files = Vec::with_capacity(rows.len());
        for row in rows {
            data_files.push(DataFileRecord::from_row(&row)?);
        }
        Ok(data_files)
    }

    pub(crate) async fn index_name_exists(
        &mut self,
        table_id: &Uuid,
        index_name: &str,
    ) -> ILResult<bool> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "index_id",
            CatalogDataType::Int64,
            false,
        )]));
        let rows = self.query_rows(&format!("SELECT index_id FROM indexlake_index WHERE table_id = {} AND index_name = '{index_name}'", self.database.sql_uuid_value(table_id)), schema).await?;
        Ok(rows.len() > 0)
    }
}

impl CatalogHelper {
    pub(crate) async fn get_namespace_id(&self, namespace_name: &str) -> ILResult<Option<Uuid>> {
        let schema = Arc::new(CatalogSchema::new(vec![Column::new(
            "namespace_id",
            CatalogDataType::Uuid,
            false,
        )]));
        let rows = self
            .query_rows(
                &format!(
                    "SELECT namespace_id FROM indexlake_namespace WHERE namespace_name = '{namespace_name}'"
                ),
                schema,
            )
            .await?;
        if rows.is_empty() {
            Ok(None)
        } else {
            let namespace_id_opt = rows[0].uuid(0)?;
            Ok(namespace_id_opt)
        }
    }

    pub(crate) async fn get_table(
        &self,
        namespace_id: &Uuid,
        table_name: &str,
    ) -> ILResult<Option<TableRecord>> {
        let schema = Arc::new(TableRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_table WHERE namespace_id = {} AND table_name = '{table_name}'",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_value(namespace_id)
                ),
                schema,
            )
            .await?;
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(TableRecord::from_row(&rows[0])?))
        }
    }

    pub(crate) async fn get_table_fields(
        &self,
        table_id: &Uuid,
    ) -> ILResult<BTreeMap<Uuid, FieldRef>> {
        let catalog_schema = Arc::new(FieldRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_field WHERE table_id = {} order by field_id asc",
                    catalog_schema
                        .select_items(self.catalog.database())
                        .join(", "),
                    self.catalog.database().sql_uuid_value(table_id)
                ),
                catalog_schema,
            )
            .await?;
        let mut field_map = BTreeMap::new();
        for row in rows {
            let field_record = FieldRecord::from_row(&row)?;
            let field_id = field_record.field_id;
            let field = field_record.into_field();
            field_map.insert(field_id, Arc::new(field));
        }
        Ok(field_map)
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
                    self.catalog.database().sql_uuid_value(table_id)
                ),
                catalog_schema,
            )
            .await?;
        let mut indexes = Vec::with_capacity(rows.len());
        for row in rows {
            indexes.push(IndexRecord::from_row(&row)?);
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
        schema: &CatalogSchemaRef,
        filters: &[Expr],
    ) -> ILResult<RowStream<'static>> {
        let mut filter_strs = filters
            .iter()
            .map(|f| f.to_sql(self.catalog.database()))
            .collect::<Result<Vec<_>, _>>()?;
        filter_strs.push(format!("{INTERNAL_FLAG_FIELD_NAME} IS NULL"));

        self.catalog
            .query(
                &format!(
                    "SELECT {} FROM {} WHERE {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    inline_row_table_name(table_id),
                    filter_strs.join(" AND ")
                ),
                Arc::clone(schema),
            )
            .await
    }

    pub(crate) async fn scan_inline_rows_by_row_ids(
        &self,
        table_id: &Uuid,
        table_schema: &CatalogSchemaRef,
        row_ids: &[i64],
    ) -> ILResult<RowStream> {
        if row_ids.is_empty() {
            return Ok(Box::pin(futures::stream::empty::<ILResult<Row>>()));
        }
        self.catalog
            .query(
                &format!(
                    "SELECT {} FROM {} WHERE {} IN ({})",
                    table_schema
                        .select_items(self.catalog.database())
                        .join(", "),
                    inline_row_table_name(table_id),
                    INTERNAL_ROW_ID_FIELD_NAME,
                    row_ids
                        .iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Arc::clone(table_schema),
            )
            .await
    }

    pub(crate) async fn get_data_files(&self, table_id: &Uuid) -> ILResult<Vec<DataFileRecord>> {
        let schema = Arc::new(DataFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_data_file WHERE table_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_value(table_id)
                ),
                schema,
            )
            .await?;
        let mut data_files = Vec::with_capacity(rows.len());
        for row in rows {
            data_files.push(DataFileRecord::from_row(&row)?);
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
                    self.catalog.database().sql_uuid_value(table_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(&row)?);
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
                    self.catalog.database().sql_uuid_value(index_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(&row)?);
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
                    self.catalog.database().sql_uuid_value(data_file_id)
                ),
                schema,
            )
            .await?;
        let mut index_files = Vec::with_capacity(rows.len());
        for row in rows {
            index_files.push(IndexFileRecord::from_row(&row)?);
        }
        Ok(index_files)
    }

    pub(crate) async fn get_index_file_by_index_id_and_data_file_id(
        &self,
        index_id: &Uuid,
        data_file_id: &Uuid,
    ) -> ILResult<Option<IndexFileRecord>> {
        let schema = Arc::new(IndexFileRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_index_file WHERE index_id = {} AND data_file_id = {}",
                    schema.select_items(self.catalog.database()).join(", "),
                    self.catalog.database().sql_uuid_value(index_id),
                    self.catalog.database().sql_uuid_value(data_file_id)
                ),
                schema,
            )
            .await?;
        if rows.is_empty() {
            Ok(None)
        } else {
            Ok(Some(IndexFileRecord::from_row(&rows[0])?))
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
                    self.catalog.database().sql_uuid_value(table_id)
                ),
                schema,
            )
            .await?;
        Ok(rows.len() > 0)
    }
}
