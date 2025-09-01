use std::collections::{HashMap, HashSet};

use uuid::Uuid;

use crate::{
    ILResult,
    catalog::{DataFileRecord, INTERNAL_FLAG_FIELD_NAME, TransactionHelper, inline_row_table_name},
    expr::Expr,
};

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
                "UPDATE {} SET {} WHERE {} AND {INTERNAL_FLAG_FIELD_NAME} NOT LIKE 'placeholder%'",
                inline_row_table_name(table_id),
                set_strs.join(", "),
                condition.to_sql(self.database)?
            ))
            .await
    }

    pub(crate) async fn update_data_file_validity(
        &mut self,
        data_file_id: &Uuid,
        validity: &[bool],
    ) -> ILResult<usize> {
        let validity_bytes = DataFileRecord::validity_to_bytes(validity);
        self.transaction
            .execute(&format!(
                "UPDATE indexlake_data_file SET validity = {} WHERE data_file_id = {}",
                self.database.sql_binary_literal(&validity_bytes),
                self.database.sql_uuid_literal(data_file_id)
            ))
            .await
    }

    pub(crate) async fn update_data_file_rows_as_invalid(
        &mut self,
        mut data_file_record: DataFileRecord,
        invalid_row_ids: &HashSet<i64>,
    ) -> ILResult<usize> {
        if invalid_row_ids.is_empty() {
            return Ok(1);
        }

        for (i, row_id) in data_file_record.row_ids.iter().enumerate() {
            if invalid_row_ids.contains(row_id) {
                data_file_record.validity[i] = false;
            }
        }

        self.update_data_file_validity(&data_file_record.data_file_id, &data_file_record.validity)
            .await
    }
}
