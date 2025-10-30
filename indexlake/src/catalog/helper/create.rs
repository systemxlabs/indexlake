use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{
    CatalogDataType, FieldRecord, INTERNAL_ROW_ID_FIELD_NAME, TransactionHelper,
    inline_row_table_name,
};

impl TransactionHelper {
    pub(crate) async fn create_inline_row_table(
        &mut self,
        table_id: &Uuid,
        field_records: &[FieldRecord],
    ) -> ILResult<()> {
        let mut columns = Vec::new();
        let pri_key_col = format!(
            "{INTERNAL_ROW_ID_FIELD_NAME} {} PRIMARY KEY",
            self.catalog
                .unparse_catalog_data_type(CatalogDataType::Uuid)
        );
        columns.push(pri_key_col);

        for field_record in field_records {
            let catalog_data_type = CatalogDataType::from_arrow(&field_record.data_type)?;
            columns.push(format!(
                "{} {} {}",
                self.catalog
                    .sql_identifier(&hex::encode(field_record.field_id)),
                self.catalog.unparse_catalog_data_type(catalog_data_type),
                if field_record.nullable {
                    "NULL"
                } else {
                    "NOT NULL"
                },
            ));
        }

        self.transaction
            .execute(&format!(
                "CREATE TABLE {} ({})",
                inline_row_table_name(table_id),
                columns.join(", ")
            ))
            .await?;
        Ok(())
    }
}
