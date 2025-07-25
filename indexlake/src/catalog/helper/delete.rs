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
    pub(crate) async fn delete_inline_rows_by_row_ids(
        &mut self,
        table_id: &Uuid,
        row_ids: &[i64],
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM {} WHERE {} IN ({})",
                inline_row_table_name(table_id),
                INTERNAL_ROW_ID_FIELD_NAME,
                row_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
            .await
    }

    pub(crate) async fn delete_inline_rows_by_condition(
        &mut self,
        table_id: &Uuid,
        condition: &Expr,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM {} WHERE {} AND {INTERNAL_FLAG_FIELD_NAME} IS NULL",
                inline_row_table_name(table_id),
                condition.to_sql(self.database)?
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
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_all_data_files(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_data_file WHERE table_id = {}",
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_table_index_files(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index_file WHERE table_id = {}",
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_index_files(&mut self, index_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index_file WHERE index_id = {}",
                self.database.sql_uuid_value(index_id)
            ))
            .await
    }

    pub(crate) async fn delete_table(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_table WHERE table_id = {}",
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_fields(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_field WHERE table_id = {}",
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_table_indexes(&mut self, table_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index WHERE table_id = {}",
                self.database.sql_uuid_value(table_id)
            ))
            .await
    }

    pub(crate) async fn delete_index(&mut self, index_id: &Uuid) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "DELETE FROM indexlake_index WHERE index_id = {}",
                self.database.sql_uuid_value(index_id)
            ))
            .await
    }
}
