use uuid::Uuid;

use crate::{
    ILResult,
    catalog::{
        INTERNAL_FLAG_FIELD_NAME, INTERNAL_ROW_ID_FIELD_NAME, TransactionHelper,
        inline_row_table_name,
    },
    expr::Expr,
};

impl TransactionHelper {
    pub(crate) async fn delete_inline_rows(
        &mut self,
        table_id: &Uuid,
        filters: &[Expr],
        row_ids: Option<&[i64]>,
    ) -> ILResult<usize> {
        if let Some(row_ids) = row_ids
            && row_ids.is_empty()
        {
            return Ok(0);
        }

        let mut filter_strs = filters
            .iter()
            .map(|f| f.to_sql(self.database))
            .collect::<Result<Vec<_>, _>>()?;
        filter_strs.push(format!(
            "{INTERNAL_FLAG_FIELD_NAME} NOT LIKE 'placeholder%'"
        ));
        if let Some(row_ids) = row_ids {
            filter_strs.push(format!(
                "{INTERNAL_ROW_ID_FIELD_NAME} IN ({})",
                row_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }

        self.transaction
            .execute(&format!(
                "DELETE FROM {} WHERE {}",
                inline_row_table_name(table_id),
                filter_strs.join(" AND ")
            ))
            .await
    }

    pub(crate) async fn delete_inline_rows_by_flag(
        &mut self,
        table_id: &Uuid,
        flag: &str,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM {} WHERE {INTERNAL_FLAG_FIELD_NAME} = '{flag}'",
                inline_row_table_name(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_dump_task(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_dump_task WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_all_data_files(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_data_file WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_table_index_files(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index_file WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_index_files(&mut self, index_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index_file WHERE index_id = {}",
                self.database.sql_uuid_literal(index_id)
            ))
            .await
    }

    pub(crate) async fn delete_table(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_table WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_fields(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_field WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_table_indexes(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index WHERE table_id = {}",
                self.database.sql_uuid_literal(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_index(&mut self, index_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index WHERE index_id = {}",
                self.database.sql_uuid_literal(index_id)
            ))
            .await
    }

    pub(crate) async fn delete_inline_indexes(&mut self, index_ids: &[Uuid]) -> ILResult<usize> {
        if index_ids.is_empty() {
            return Ok(0);
        }

        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_inline_index WHERE index_id IN ({})",
                index_ids
                    .iter()
                    .map(|id| self.database.sql_uuid_literal(id))
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .await
    }
}
