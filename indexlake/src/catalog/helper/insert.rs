use std::sync::Arc;

use arrow::array::{LargeBinaryArray, RecordBatch};
use uuid::Uuid;

use crate::catalog::{
    DataFileRecord, FieldRecord, IndexFileRecord, IndexRecord, InlineIndexRecord, TableRecord,
    TaskRecord, TransactionHelper, inline_row_table_name,
};
use crate::utils::build_row_id_array;
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

    pub(crate) async fn insert_task(&mut self, task: TaskRecord) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "INSERT INTO indexlake_task ({}) VALUES {}",
                TaskRecord::catalog_schema()
                    .select_items(self.catalog.as_ref())
                    .join(", "),
                task.to_sql(self.catalog.as_ref())
            ))
            .await
    }

    pub(crate) async fn insert_inline_rows(
        &mut self,
        table_id: &Uuid,
        field_names: &[String],
        batches: &[RecordBatch],
    ) -> ILResult<()> {
        if batches.is_empty() {
            return Ok(());
        }
        if field_names.len() != batches[0].columns().len() {
            return Err(ILError::internal(format!(
                "field_names and batches must have the same length, got {} and {}",
                field_names.len(),
                batches[0].columns().len()
            )));
        }
        let num_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        if num_rows == 0 {
            return Ok(());
        }

        let table_name = inline_row_table_name(table_id);
        let quoted_field_names: Vec<String> = field_names
            .iter()
            .map(|name| self.catalog.sql_identifier(name))
            .collect();

        self.transaction
            .insert_rows(&table_name, &quoted_field_names, batches)
            .await?;
        Ok(())
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
    ) -> ILResult<()> {
        if inline_indexes.is_empty() {
            return Ok(());
        }
        let index_id_arr = build_row_id_array(inline_indexes.iter().map(|r| r.index_id))?;
        let index_data_arr =
            LargeBinaryArray::from_iter_values(inline_indexes.iter().map(|r| &r.index_data));

        let batch = RecordBatch::try_new(
            InlineIndexRecord::arrow_schema(),
            vec![Arc::new(index_id_arr), Arc::new(index_data_arr)],
        )?;

        self.transaction
            .insert_rows(
                "indexlake_inline_index",
                &["index_id".to_string(), "index_data".to_string()],
                &[batch],
            )
            .await?;
        Ok(())
    }
}
