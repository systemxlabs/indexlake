use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::catalog::{RowValidity, TransactionHelper, inline_row_table_name};
use crate::expr::Expr;
use crate::{ILError, ILResult};

impl TransactionHelper {
    pub(crate) async fn update_namespace_name(
        &mut self,
        namespace_id: &Uuid,
        namespace_name: &str,
    ) -> ILResult<usize> {
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_namespace SET namespace_name = {} WHERE namespace_id = {}",
                self.catalog.sql_string_literal(namespace_name),
                self.catalog.sql_uuid_literal(namespace_id),
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
                self.catalog.sql_string_literal(table_name),
                self.catalog.sql_uuid_literal(table_id),
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
                self.catalog.sql_string_literal(field_name),
                self.catalog.sql_uuid_literal(field_id),
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
                self.catalog.sql_identifier(field_name),
                self.catalog.unparse_expr(new_value)?,
            ));
        }

        self.transaction
            .execute(&format!(
                "UPDATE {} SET {} WHERE {}",
                inline_row_table_name(table_id),
                set_strs.join(", "),
                self.catalog.unparse_expr(condition)?
            ))
            .await
    }

    pub(crate) async fn update_data_file_validity(
        &mut self,
        data_file_id: &Uuid,
        old_validity: &RowValidity,
        new_validity: &RowValidity,
    ) -> ILResult<usize> {
        let valid_record_count = new_validity.count_valid() as i64;
        let update_count = self.transaction
            .execute(&format!(
                "UPDATE indexlake_data_file SET validity = {}, valid_record_count = {} WHERE data_file_id = {} AND validity = {}",
                self.catalog.sql_binary_literal(new_validity.bytes()),
                valid_record_count,
                self.catalog.sql_uuid_literal(data_file_id),
                self.catalog.sql_binary_literal(old_validity.bytes()),
            ))
            .await?;
        if update_count != 1 {
            return Err(ILError::internal(
                "Failed to update data file validity due to concurrency conflict",
            ));
        }
        Ok(update_count)
    }

    pub(crate) async fn update_data_file_rows_as_invalid(
        &mut self,
        data_file_id: &Uuid,
        mut validity: RowValidity,
        sorted_row_ids: &[Uuid],
        invalid_row_ids: &HashSet<Uuid>,
    ) -> ILResult<usize> {
        debug_assert!(sorted_row_ids.is_sorted());

        if invalid_row_ids.is_empty() {
            return Ok(1);
        }

        let old_validity = validity.clone();

        for invalid_row_id in invalid_row_ids {
            if let Ok(idx) = sorted_row_ids.binary_search(invalid_row_id) {
                validity.set(idx, false);
            }
        }

        self.update_data_file_validity(data_file_id, &old_validity, &validity)
            .await
    }
}
