use uuid::Uuid;

use crate::catalog::{
    DataFileRecord, FieldRecord, IndexFileRecord, IndexRecord, InlineIndexRecord, TableRecord,
    TransactionHelper, inline_row_table_name,
};
use crate::{ILError, ILResult};

impl TransactionHelper {
    pub(crate) async fn insert_namespace(
        &mut self,
        namespace_id: &Uuid,
        namespace_name: &str,
    ) -> ILResult<()> {
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_namespace (namespace_id, namespace_name) VALUES ({}, '{namespace_name}')",
                self.catalog.sql_uuid_literal(namespace_id)
            ))
            .await?;
        Ok(())
    }

    pub(crate) async fn insert_table(&mut self, table_record: &TableRecord) -> ILResult<()> {
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_table ({}) VALUES {}",
                TableRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                table_record.to_sql(self.catalog.as_ref())?
            ))
            .await?;
        Ok(())
    }

    pub(crate) async fn insert_fields(&mut self, fields: &[FieldRecord]) -> ILResult<()> {
        if fields.is_empty() {
            return Ok(());
        }
        let mut values = Vec::new();
        for record in fields {
            values.push(record.to_sql(self.catalog.as_ref())?);
        }
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_field ({}) VALUES {}",
                FieldRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                values.join(", ")
            ))
            .await?;
        Ok(())
    }

    pub(crate) async fn insert_inline_rows(
        &mut self,
        table_id: &Uuid,
        field_names: &[String],
        values: Vec<Vec<String>>,
    ) -> ILResult<()> {
        if values.is_empty() {
            return Ok(());
        }
        if field_names.len() != values.len() {
            return Err(ILError::internal(format!(
                "field_names and values must have the same length, got {} and {}",
                field_names.len(),
                values.len()
            )));
        }

        let num_columns = field_names.len();
        let num_rows = values[0].len();

        if num_rows == 0 {
            return Ok(());
        }

        let mut values_strs = Vec::with_capacity(num_rows);
        for row_idx in 0..num_rows {
            let mut row_values = Vec::with_capacity(num_columns);
            for col in values.iter() {
                row_values.push(col[row_idx].as_str());
            }
            values_strs.push(format!("({})", row_values.join(",")));
        }

        self.transaction
            .execute(&format!(
                "INSERT INTO {} ({}) VALUES {}",
                inline_row_table_name(table_id),
                field_names
                    .iter()
                    .map(|name| self.catalog.sql_identifier(name))
                    .collect::<Vec<_>>()
                    .join(", "),
                values_strs.join(", ")
            ))
            .await?;
        Ok(())
    }

    pub(crate) async fn insert_task(&mut self, task_id: &str) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_task VALUES ({})",
                self.catalog.sql_string_literal(task_id)
            ))
            .await
    }

    pub(crate) async fn insert_data_files(
        &mut self,
        data_files: &[DataFileRecord],
    ) -> ILResult<usize> {
        if data_files.is_empty() {
            return Ok(0);
        }
        let values = data_files
            .iter()
            .map(|r| r.to_sql(self.catalog.as_ref()))
            .collect::<Vec<_>>();
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_data_file ({}) VALUES {}",
                DataFileRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                values.join(", ")
            ))
            .await
    }

    pub(crate) async fn insert_index(&mut self, index_record: &IndexRecord) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_index ({}) VALUES {}",
                IndexRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                index_record.to_sql(self.catalog.as_ref())
            ))
            .await
    }

    pub(crate) async fn insert_index_files(
        &mut self,
        index_files: &[IndexFileRecord],
    ) -> ILResult<usize> {
        if index_files.is_empty() {
            return Ok(0);
        }
        let values = index_files
            .iter()
            .map(|r| r.to_sql(self.catalog.as_ref()))
            .collect::<Vec<_>>();
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_index_file ({}) VALUES {}",
                IndexFileRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                values.join(", ")
            ))
            .await
    }

    pub(crate) async fn insert_inline_indexes(
        &mut self,
        inline_indexes: &[InlineIndexRecord],
    ) -> ILResult<usize> {
        if inline_indexes.is_empty() {
            return Ok(0);
        }
        let values = inline_indexes
            .iter()
            .map(|r| r.to_sql(self.catalog.as_ref()))
            .collect::<Vec<_>>();
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_inline_index ({}) VALUES {}",
                InlineIndexRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                values.join(", ")
            ))
            .await
    }
}
