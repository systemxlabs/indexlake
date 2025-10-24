use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{
    CatalogDataType, CatalogDatabase, FieldRecord, INTERNAL_ROW_ID_FIELD_NAME, TransactionHelper,
    inline_row_table_name,
};

impl TransactionHelper {
    pub(crate) async fn create_inline_row_table(
        &mut self,
        table_id: &Uuid,
        field_records: &[FieldRecord],
    ) -> ILResult<()> {
        let mut columns = Vec::new();
        match self.database {
            CatalogDatabase::Postgres => {
                columns.push(format!("{INTERNAL_ROW_ID_FIELD_NAME} UUID PRIMARY KEY"));
            }
            CatalogDatabase::Sqlite => {
                columns.push(format!("{INTERNAL_ROW_ID_FIELD_NAME} BLOB PRIMARY KEY"));
            }
        }

        for field_record in field_records {
            columns.push(format!(
                "{} {} {}",
                self.database
                    .sql_identifier(&hex::encode(field_record.field_id)),
                CatalogDataType::from_arrow(&field_record.data_type)?.to_sql(self.database),
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
