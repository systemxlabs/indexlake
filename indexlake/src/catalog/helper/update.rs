use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{DataFileRecord, RowValidity, TransactionHelper, inline_row_table_name};
use crate::expr::Expr;

impl TransactionHelper {
    pub(crate) async fn update_namespace_name(
        &mut self,
        namespace_id: &Uuid,
        namespace_name: &str,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_namespace SET namespace_name = {} WHERE namespace_id = {}",
                self.database.sql_string_literal(namespace_name),
                self.database.sql_uuid_literal(namespace_id),
            ))
            .await
    }

    pub(crate) async fn update_table_name(
        &mut self,
        table_id: &Uuid,
        table_name: &str,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_table SET table_name = {} WHERE table_id = {}",
                self.database.sql_string_literal(table_name),
                self.database.sql_uuid_literal(table_id),
            ))
            .await
    }

    pub(crate) async fn update_field_name(
        &mut self,
        field_id: &Uuid,
        field_name: &str,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_field SET field_name = {} WHERE field_id = {}",
                self.database.sql_string_literal(field_name),
                self.database.sql_uuid_literal(field_id),
            ))
            .await
    }

    pub(crate) async fn update_inline_rows(
        &mut self,
        table_id: &Uuid,
        set_map: &HashMap<String, Expr>,
        condition: &Expr,
    ) -> ILResult<usize> {
        let mut set_strs = Vec::new();
        for (field_name, new_value) in set_map {
            set_strs.push(format!(
                "{} = {}",
                self.database.sql_identifier(field_name),
                new_value.to_sql(self.database)?,
            ));
        }

        self.transaction
            .execute(&format!(
                "UPDATE {} SET {} WHERE {}",
                inline_row_table_name(table_id),
                set_strs.join(", "),
                condition.to_sql(self.database)?
            ))
            .await
    }

    pub(crate) async fn update_data_file_validity(
        &mut self,
        data_file_id: &Uuid,
        validity: &RowValidity,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_data_file SET validity = {} WHERE data_file_id = {}",
                self.database.sql_binary_literal(validity.bytes()),
                self.database.sql_uuid_literal(data_file_id)
            ))
            .await
    }

    pub(crate) async fn update_data_file_rows_as_invalid(
        &mut self,
        mut data_file_record: DataFileRecord,
        sorted_row_ids: &[Uuid],
        invalid_row_ids: &HashSet<Uuid>,
    ) -> ILResult<usize> {
        debug_assert!(sorted_row_ids.is_sorted());

        if invalid_row_ids.is_empty() {
            return Ok(1);
        }

        for invalid_row_id in invalid_row_ids {
            if let Ok(idx) = sorted_row_ids.binary_search(invalid_row_id) {
                data_file_record.validity.set(idx, false);
            }
        }

        self.update_data_file_validity(&data_file_record.data_file_id, &data_file_record.validity)
            .await
    }
}
