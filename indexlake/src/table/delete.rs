use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
use futures::StreamExt;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::ILResult;
use crate::catalog::{
    CatalogSchema, DataFileRecord, INTERNAL_ROW_ID_FIELD_REF, TransactionHelper,
    rows_to_record_batch,
};
use crate::expr::Expr;
use crate::storage::{
    Storage, find_matched_row_ids_from_data_file, read_row_id_array_from_data_file,
};
use crate::table::{Table, TableSchemaRef};
use crate::utils::fixed_size_binary_array_to_uuids;

pub(crate) async fn process_delete_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    condition: &Expr,
    matched_data_file_row_ids: HashMap<Uuid, HashSet<Uuid>>,
) -> ILResult<usize> {
    // Directly delete inline rows
    let inline_delete_count =
        delete_inline_rows(tx_helper, &table.table_id, &table.table_schema, condition).await?;

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;

    let mut file_delete_count = 0;
    for data_file_record in data_file_records {
        let row_id_array = read_row_id_array_from_data_file(
            table.storage.as_ref(),
            &data_file_record.relative_path,
            data_file_record.format,
        )
        .await?;
        let row_ids = fixed_size_binary_array_to_uuids(&row_id_array)?;

        let delete_count = if let Some(matched_row_ids) =
            matched_data_file_row_ids.get(&data_file_record.data_file_id)
        {
            tx_helper
                .update_data_file_rows_as_invalid(data_file_record, &row_ids, matched_row_ids)
                .await?;
            matched_row_ids.len()
        } else {
            delete_data_file_rows_by_condition(
                tx_helper,
                table,
                condition,
                data_file_record,
                &row_ids,
            )
            .await?
        };

        file_delete_count += delete_count;
    }
    Ok(inline_delete_count + file_delete_count)
}

pub(crate) async fn delete_inline_rows(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    table_schema: &TableSchemaRef,
    condition: &Expr,
) -> ILResult<usize> {
    // TODO improve performance through projection
    if tx_helper
        .catalog
        .supports_filter(condition, &table_schema.arrow_schema)?
    {
        tx_helper
            .delete_inline_rows(table_id, std::slice::from_ref(condition), None)
            .await
    } else {
        let catalog_schema = Arc::new(CatalogSchema::from_arrow(&table_schema.arrow_schema)?);
        let row_stream = tx_helper
            .scan_inline_rows(table_id, &catalog_schema, &[], None, None)
            .await?;
        let mut chunk_stream = row_stream.chunks(100);
        let mut matched_row_ids = Vec::new();
        while let Some(row_chunk) = chunk_stream.next().await {
            let rows = row_chunk.into_iter().collect::<ILResult<Vec<_>>>()?;
            let record_batch = rows_to_record_batch(&table_schema.arrow_schema, &rows)?;
            let bool_array = condition.condition_eval(&record_batch)?;
            for (i, v) in bool_array.iter().enumerate() {
                if let Some(v) = v
                    && v
                {
                    matched_row_ids.push(rows[i].uuid(0)?.expect("row_id is not null"));
                }
            }
        }
        drop(chunk_stream);
        tx_helper
            .delete_inline_rows(table_id, &[], Some(matched_row_ids.as_slice()))
            .await
    }
}

pub(crate) async fn delete_data_file_rows_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    condition: &Expr,
    data_file_record: DataFileRecord,
    row_ids: &[Uuid],
) -> ILResult<usize> {
    let deleted_row_ids = find_matched_row_ids_from_data_file(
        table.storage.as_ref(),
        &table.table_schema,
        condition,
        &data_file_record,
    )
    .await?;

    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, row_ids, &deleted_row_ids)
        .await?;
    Ok(deleted_row_ids.len())
}

pub(crate) async fn process_delete_by_row_id_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    row_id_condition: &Expr,
) -> ILResult<usize> {
    let inline_delete_count = delete_inline_rows(
        tx_helper,
        &table.table_id,
        &table.table_schema,
        row_id_condition,
    )
    .await?;

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;

    let mut file_delete_count = 0;
    for mut data_file_record in data_file_records {
        let row_id_array = read_row_id_array_from_data_file(
            table.storage.as_ref(),
            &data_file_record.relative_path,
            data_file_record.format,
        )
        .await?;

        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![INTERNAL_ROW_ID_FIELD_REF.clone()])),
            vec![Arc::new(row_id_array)],
        )?;
        let bool_array = row_id_condition.condition_eval(&batch)?;

        let mut delete_count = 0;
        for (i, v) in bool_array.iter().enumerate() {
            if let Some(v) = v
                && v
            {
                data_file_record.validity.set(i, false);
                delete_count += 1;
            }
        }
        file_delete_count += delete_count;

        tx_helper
            .update_data_file_validity(&data_file_record.data_file_id, &data_file_record.validity)
            .await?;
    }
    Ok(inline_delete_count + file_delete_count)
}

pub(crate) async fn parallel_find_matched_data_file_row_ids(
    storage: Arc<dyn Storage>,
    table_schema: TableSchemaRef,
    condition: Expr,
    data_file_records: Vec<DataFileRecord>,
) -> ILResult<HashMap<Uuid, HashSet<Uuid>>> {
    condition.check_data_type(&table_schema.arrow_schema, &DataType::Boolean)?;

    let mut handles = Vec::new();
    for data_file_record in data_file_records {
        let storage = storage.clone();
        let table_schema = table_schema.clone();
        let condition = condition.clone();

        let handle: JoinHandle<ILResult<(Uuid, HashSet<Uuid>)>> = tokio::spawn(async move {
            let matched_row_ids = find_matched_row_ids_from_data_file(
                storage.as_ref(),
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
