use arrow_schema::{FieldRef, Schema};
use uuid::Uuid;

use crate::{
    ILError, ILResult,
    catalog::{FieldRecord, Scalar, TransactionHelper},
    check_schema_contains_system_column,
    table::{Table, check_default_value},
};

#[derive(Debug)]
pub enum TableAlter {
    RenameTable {
        new_name: String,
    },
    RenameColumn {
        old_name: String,
        new_name: String,
    },
    AddColumn {
        field: FieldRef,
        default_value: Scalar,
    },
}

pub(crate) async fn process_table_alter(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    alter: TableAlter,
) -> ILResult<()> {
    match alter {
        TableAlter::RenameTable { new_name } => {
            tx_helper
                .update_table_name(&table.table_id, &new_name)
                .await?;
        }
        TableAlter::RenameColumn { old_name, new_name } => {
            alter_rename_column(tx_helper, &table.table_id, &old_name, &new_name).await?;
        }
        TableAlter::AddColumn {
            field,
            default_value,
        } => {
            alter_add_column(tx_helper, &table.table_id, field, default_value).await?;
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

pub(crate) async fn alter_add_column(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    field: FieldRef,
    default_value: Scalar,
) -> ILResult<()> {
    check_default_value(field.as_ref(), &default_value)?;

    let schema = Schema::new(vec![field.clone()]);
    check_schema_contains_system_column(&schema)?;

    if tx_helper
        .get_table_field(table_id, field.name())
        .await?
        .is_some()
    {
        return Err(ILError::invalid_input(format!(
            "Field name {} already exists for table id {}",
            field.name(),
            table_id
        )));
    }

    let field_id = Uuid::now_v7();
    let field_record = FieldRecord::new(field_id, *table_id, field.as_ref(), Some(default_value));
    tx_helper.insert_fields(&[field_record]).await?;
    Ok(())
}
