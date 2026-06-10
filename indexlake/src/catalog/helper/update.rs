use std::collections::{HashMap, HashSet};

use arrow::array::ArrayRef;
use uuid::Uuid;

use crate::catalog::{
    INTERNAL_ROW_ID_FIELD_NAME, InlineIndexRecord, RowValidity, Scalar, TransactionHelper,
    inline_row_table_name,
};
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

    pub(crate) async fn update_inline_rows_by_row_ids(
        &mut self,
        table_id: &Uuid,
        row_ids: &[Uuid],
        column_names: &[String],
        column_arrays: &[ArrayRef],
    ) -> ILResult<usize> {
        let mut total_updated = 0;
        for (i, row_id) in row_ids.iter().enumerate() {
            let mut set_strs = Vec::new();
            for (name, array) in column_names.iter().zip(column_arrays.iter()) {
                let scalar = Scalar::try_from_array(array.as_ref(), i)?;
                set_strs.push(format!(
                    "{} = {}",
                    self.catalog.sql_identifier(name),
                    scalar.to_sql(self.catalog.as_ref())?
                ));
            }
            total_updated += self
                .transaction
                .execute(&format!(
                    "UPDATE {} SET {} WHERE {} = {}",
                    inline_row_table_name(table_id),
                    set_strs.join(", "),
                    INTERNAL_ROW_ID_FIELD_NAME,
                    self.catalog.sql_uuid_literal(row_id)
                ))
                .await?;
        }
        Ok(total_updated)
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

    /// Mark specific row_ids as invalid in inline index segments.
    /// Returns the number of segments updated.
    pub(crate) async fn invalidate_inline_index_rows(
        &mut self,
        index_id: &Uuid,
        invalid_row_ids: &HashSet<Uuid>,
    ) -> ILResult<usize> {
        if invalid_row_ids.is_empty() {
            return Ok(0);
        }

        // Fetch all inline index records for this index
        let schema = std::sync::Arc::new(InlineIndexRecord::catalog_schema());
        let rows = self
            .query_rows(
                &format!(
                    "SELECT {} FROM indexlake_inline_index WHERE index_id = {} ORDER BY created_at ASC",
                    schema.select_items(self.catalog.as_ref()).join(", "),
                    self.catalog.sql_uuid_literal(index_id),
                ),
                schema,
            )
            .await?;

        let mut updated_count = 0;

        for row in rows {
            let record = InlineIndexRecord::from_row(row)?;
            let mut modified = false;
            let mut validity = record.validity.clone();

            for (i, row_id) in record.row_ids.iter().enumerate() {
                if invalid_row_ids.contains(row_id) && validity.is_valid(i) {
                    validity.set(i, false);
                    modified = true;
                }
            }

            if modified {
                // Update the record in the database
                let update_count = self
                    .transaction
                    .execute(&format!(
                        "UPDATE indexlake_inline_index SET validity = {} WHERE index_id = {} AND created_at = {}",
                        self.catalog.sql_binary_literal(validity.bytes()),
                        self.catalog.sql_uuid_literal(&record.index_id),
                        record.created_at,
                    ))
                    .await?;
                updated_count += update_count;
            }
        }

        Ok(updated_count)
    }
}
