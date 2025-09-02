use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::{
    array::{RecordBatch, RecordBatchOptions},
    datatypes::SchemaRef,
};
use futures::StreamExt;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::{
    ILResult, RecordBatchStream,
    catalog::{DataFileRecord, TransactionHelper},
    expr::Expr,
    storage::{Storage, read_data_file_by_record, read_data_file_by_record_and_row_id_condition},
    table::{Table, process_insert_into_inline_rows, rebuild_inline_indexes},
    utils::extract_row_ids_from_record_batch,
};

pub(crate) async fn process_update_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    set_map: HashMap<String, Expr>,
    condition: &Expr,
    mut matched_data_file_rows: HashMap<Uuid, RecordBatchStream>,
) -> ILResult<()> {
    let updated_row_count = tx_helper
        .update_inline_rows(&table.table_id, &set_map, condition)
        .await?;

    if updated_row_count != 0 {
        rebuild_inline_indexes(
            tx_helper,
            &table.table_id,
            &table.schema,
            &table.index_manager,
        )
        .await?;
    }

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;

    for data_file_record in data_file_records {
        if let Some(stream) = matched_data_file_rows.get_mut(&data_file_record.data_file_id) {
            update_data_file_rows_by_matched_rows(
                tx_helper,
                table,
                &set_map,
                stream,
                data_file_record,
            )
            .await?;
        } else {
            update_data_file_rows_by_condition(
                tx_helper,
                table,
                &set_map,
                condition,
                data_file_record,
            )
            .await?;
        }
    }

    Ok(())
}

pub(crate) async fn update_data_file_rows_by_matched_rows(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    set_map: &HashMap<String, Expr>,
    matched_data_file_rows: &mut RecordBatchStream,
    data_file_record: DataFileRecord,
) -> ILResult<()> {
    let mut updated_row_ids = HashSet::new();
    while let Some(batch) = matched_data_file_rows.next().await {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let row_ids = extract_row_ids_from_record_batch(&batch)?;
        updated_row_ids.extend(row_ids);
        let updated_batch = update_record_batch(&batch, set_map)?;
        process_insert_into_inline_rows(tx_helper, table, &[updated_batch]).await?;
    }
    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, &updated_row_ids)
        .await?;
    Ok(())
}

pub(crate) async fn update_data_file_rows_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    set_map: &HashMap<String, Expr>,
    condition: &Expr,
    data_file_record: DataFileRecord,
) -> ILResult<()> {
    let mut stream = if condition.only_visit_row_id_column() {
        read_data_file_by_record_and_row_id_condition(
            &table.storage,
            &table.schema,
            &data_file_record,
            None,
            condition,
        )
        .await?
    } else {
        read_data_file_by_record(
            &table.storage,
            &table.schema,
            &data_file_record,
            None,
            vec![condition.clone()],
            None,
            1024,
        )
        .await?
    };

    let mut updated_row_ids = HashSet::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;

        let row_ids = extract_row_ids_from_record_batch(&batch)?;
        updated_row_ids.extend(row_ids);
        let updated_batch = update_record_batch(&batch, set_map)?;
        process_insert_into_inline_rows(tx_helper, table, &[updated_batch]).await?;
    }

    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, &updated_row_ids)
        .await?;
    Ok(())
}

pub(crate) async fn parallel_find_matched_data_file_rows(
    storage: Arc<Storage>,
    table_schema: SchemaRef,
    condition: Expr,
    data_file_records: Vec<DataFileRecord>,
) -> ILResult<HashMap<Uuid, RecordBatchStream>> {
    let mut handles = Vec::new();
    for data_file_record in data_file_records {
        let storage = storage.clone();
        let table_schema = table_schema.clone();
        let condition = condition.clone();
        let handle: JoinHandle<ILResult<(Uuid, RecordBatchStream)>> = tokio::spawn(async move {
            let mut stream = if condition.only_visit_row_id_column() {
                read_data_file_by_record_and_row_id_condition(
                    &storage,
                    &table_schema,
                    &data_file_record,
                    None,
                    &condition,
                )
                .await?
            } else {
                read_data_file_by_record(
                    &storage,
                    &table_schema,
                    &data_file_record,
                    None,
                    vec![condition],
                    None,
                    1024,
                )
                .await?
            };

            // prefetch record batch into memory
            let mut prefetch_row_count = 0;
            let mut stream_exhausted = true;
            let mut batches = Vec::new();
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                prefetch_row_count += batch.num_rows();
                batches.push(batch);
                if prefetch_row_count > 1000 {
                    stream_exhausted = false;
                    break;
                }
            }

            let memory_stream =
                Box::pin(futures::stream::iter(batches).map(Ok)) as RecordBatchStream;

            let merged_stream = if stream_exhausted {
                memory_stream
            } else {
                Box::pin(futures::stream::select_all(vec![memory_stream, stream]))
            };
            Ok((data_file_record.data_file_id, merged_stream))
        });
        handles.push(handle);
    }
    let mut matched_rows = HashMap::new();
    for handle in handles {
        let (data_file_id, stream) = handle.await??;
        matched_rows.insert(data_file_id, stream);
    }
    Ok(matched_rows)
}

fn update_record_batch(
    batch: &RecordBatch,
    set_map: &HashMap<String, Expr>,
) -> ILResult<RecordBatch> {
    let mut columns = batch.columns().to_vec();
    for (name, value) in set_map {
        let idx = batch.schema().index_of(name)?;
        let new_array = value.eval(batch)?.into_array(batch.num_rows())?;
        columns[idx] = Arc::new(new_array);
    }
    let options = RecordBatchOptions::default().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        batch.schema(),
        columns,
        &options,
    )?)
}
