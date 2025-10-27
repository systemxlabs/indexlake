use uuid::Uuid;

use crate::{ILError, ILResult, catalog::TransactionHelper, table::Table};

#[derive(Debug)]
pub enum TableAlter {
    RenameColumn { old_name: String, new_name: String },
}

pub(crate) async fn process_table_alter(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    alter: TableAlter,
) -> ILResult<()> {
    match alter {
        TableAlter::RenameColumn { old_name, new_name } => {
            alter_rename_column(tx_helper, &table.table_id, &old_name, &new_name).await?;
        }
    }
    Ok(())
}

pub(crate) async fn alter_rename_column(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    old_name: &str,
    new_name: &str,
) -> ILResult<()> {
    let Some(field_record) = tx_helper.get_table_field(table_id, old_name).await? else {
        return Err(ILError::invalid_input(format!(
            "Field name {old_name} not found for table id {table_id}"
        )));
    };
    tx_helper
        .update_field_name(&field_record.field_id, new_name)
        .await?;
    Ok(())
}
