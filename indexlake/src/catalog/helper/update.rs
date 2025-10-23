use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{DataFileRecord, RowValidity, TransactionHelper, inline_row_table_name};
use crate::expr::Expr;

impl TransactionHelper {
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
        row_ids: &[Uuid],
        invalid_row_ids: &HashSet<Uuid>,
    ) -> ILResult<usize> {
        if invalid_row_ids.is_empty() {
            return Ok(1);
        }

        // TODO accelerate this by binary search
        for (i, row_id) in row_ids.iter().enumerate() {
            if invalid_row_ids.contains(row_id) {
                data_file_record.validity.set(i, false);
            }
        }

        self.update_data_file_validity(&data_file_record.data_file_id, &data_file_record.validity)
            .await
    }
}
