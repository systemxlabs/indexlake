use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{FixedSizeBinaryArray, RecordBatch};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{DataFileRecord, INTERNAL_ROW_ID_FIELD_REF, TransactionHelper};
use crate::expr::Expr;
use crate::storage::{Storage, find_matched_row_ids_from_data_file};
use crate::table::Table;

pub(crate) async fn process_delete_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    condition: &Expr,
    matched_data_file_row_ids: HashMap<Uuid, HashSet<Uuid>>,
) -> ILResult<()> {
    // Directly delete inline rows
    tx_helper
        .delete_inline_rows(&table.table_id, std::slice::from_ref(condition), None)
        .await?;

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;
    for data_file_record in data_file_records {
        if let Some(matched_row_ids) = matched_data_file_row_ids.get(&data_file_record.data_file_id)
        {
            tx_helper
                .update_data_file_rows_as_invalid(data_file_record, matched_row_ids)
                .await?;
        } else {
            delete_data_file_rows_by_condition(tx_helper, table, condition, data_file_record)
                .await?;
        }
    }
    Ok(())
}

pub(crate) async fn delete_data_file_rows_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    condition: &Expr,
    data_file_record: DataFileRecord,
) -> ILResult<()> {
    let deleted_row_ids = find_matched_row_ids_from_data_file(
        &table.storage,
        &table.schema,
        condition,
        &data_file_record,
    )
    .await?;

    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, &deleted_row_ids)
        .await?;
    Ok(())
}

pub(crate) async fn process_delete_by_row_id_condition(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    row_id_condition: &Expr,
) -> ILResult<()> {
    tx_helper
        .delete_inline_rows(table_id, std::slice::from_ref(row_id_condition), None)
        .await?;

    let data_file_records = tx_helper.get_data_files(table_id).await?;
    for mut data_file_record in data_file_records {
        // We need row index to update validity, so we need to get all row ids
        let row_ids = data_file_record.row_ids;
        let row_id_array = FixedSizeBinaryArray::try_from_iter(row_ids.into_iter())?;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![INTERNAL_ROW_ID_FIELD_REF.clone()])),
            vec![Arc::new(row_id_array)],
        )?;
        let bool_array = row_id_condition.condition_eval(&batch)?;

        for (i, v) in bool_array.iter().enumerate() {
            if let Some(v) = v
                && v
            {
                data_file_record.validity[i] = false;
            }
        }

        tx_helper
            .update_data_file_validity(&data_file_record.data_file_id, &data_file_record.validity)
            .await?;
    }
    Ok(())
}

pub(crate) async fn parallel_find_matched_data_file_row_ids(
    storage: Arc<Storage>,
    table_schema: SchemaRef,
    condition: Expr,
    data_file_records: Vec<DataFileRecord>,
) -> ILResult<HashMap<Uuid, HashSet<Uuid>>> {
    condition.check_data_type(&table_schema, &DataType::Boolean)?;

    let mut handles = Vec::new();
    for data_file_record in data_file_records {
        let storage = storage.clone();
        let table_schema = table_schema.clone();
        let condition = condition.clone();

        let handle: JoinHandle<ILResult<(Uuid, HashSet<Uuid>)>> = tokio::spawn(async move {
            let matched_row_ids = find_matched_row_ids_from_data_file(
                &storage,
                &table_schema,
                &condition,
                &data_file_record,
            )
            .await?;
            Ok((data_file_record.data_file_id, matched_row_ids))
        });
        handles.push(handle);
    }

    let mut matched_row_ids = HashMap::new();
    for handle in handles {
        let (data_file_id, row_ids) = handle.await??;
        matched_row_ids.insert(data_file_id, row_ids);
    }
    Ok(matched_row_ids)
}
