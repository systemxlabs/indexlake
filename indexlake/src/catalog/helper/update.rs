use std::collections::{HashMap, HashSet};

use arrow::array::ArrayRef;
use uuid::Uuid;

use crate::catalog::{
    CatalogDataType, CatalogSchema, Column, INTERNAL_ROW_ID_FIELD_NAME, RowValidity, Scalar,
    TransactionHelper, inline_row_table_name,
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

        // Fetch only the columns we need to invalidate rows
        let schema = std::sync::Arc::new(CatalogSchema::new(vec![
            Column::new("index_id", CatalogDataType::Uuid, false),
            Column::new("created_at", CatalogDataType::Int64, false),
            Column::new("row_ids", CatalogDataType::Binary, false),
            Column::new("validity", CatalogDataType::Binary, false),
        ]));
        let rows = self
            .query_rows(
                &format!(
                    "SELECT index_id, created_at, row_ids, validity FROM indexlake_inline_index WHERE index_id = {} ORDER BY created_at ASC",
                    self.catalog.sql_uuid_literal(index_id),
                ),
                schema,
            )
            .await?;

        let mut updated_count = 0;

        for mut row in rows {
            let idx_id = row.uuid(0)?.expect("index_id is not null");
            let idx_created_at = row.int64(1)?.expect("created_at is not null");
            let row_ids_bytes = row.binary_owned(2)?.expect("row_ids is not null");
            let row_ids = crate::utils::deserialize_row_ids(&row_ids_bytes)?;
            let validity_bytes = row.binary_owned(3)?.expect("validity is not null");
            let mut validity = RowValidity::from(validity_bytes, row_ids.len());
            let mut modified = false;

            for (i, row_id) in row_ids.iter().enumerate() {
                if invalid_row_ids.contains(row_id) && validity.is_valid(i)? {
                    validity.set(i, false);
                    modified = true;
                }
            }

            if modified {
                if validity.count_valid() == 0 {
                    // All rows invalid: delete the record entirely
                    let delete_count = self
                        .transaction
                        .execute(&format!(
                            "DELETE FROM indexlake_inline_index WHERE index_id = {} AND created_at = {}",
                            self.catalog.sql_uuid_literal(&idx_id),
                            idx_created_at,
                        ))
                        .await?;
                    updated_count += delete_count;
                } else {
                    let update_count = self
                        .transaction
                        .execute(&format!(
                            "UPDATE indexlake_inline_index SET validity = {} WHERE index_id = {} AND created_at = {}",
                            self.catalog.sql_binary_literal(validity.bytes()),
                            self.catalog.sql_uuid_literal(&idx_id),
                            idx_created_at,
                        ))
                        .await?;
                    updated_count += update_count;
                }
            }
        }

        Ok(updated_count)
    }
}
