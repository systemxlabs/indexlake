use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions};
use futures::StreamExt;
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::catalog::{CatalogSchema, DataFileRecord, TransactionHelper, rows_to_record_batch};
use crate::expr::Expr;
use crate::storage::{Storage, read_data_file_by_record, read_row_id_array_from_data_file};
use crate::table::{
    Table, TableSchemaRef, process_insert_into_inline_rows, rebuild_inline_indexes,
};
use crate::utils::{extract_row_ids_from_record_batch, fixed_size_binary_array_to_uuids};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug)]
pub struct TableUpdate {
    pub set_map: HashMap<String, Expr>,
    pub condition: Expr,
}

impl TableUpdate {
    pub(crate) fn rewrite_columns(
        mut self,
        field_name_id_map: &HashMap<String, Uuid>,
    ) -> ILResult<Self> {
        let mut new_set_map = HashMap::with_capacity(self.set_map.len());
        for (key, value) in self.set_map {
            let Some(field_id) = field_name_id_map.get(&key) else {
                return Err(ILError::invalid_input(format!("Not found field {key}")));
            };
            let new_value = value.rewrite_columns(field_name_id_map)?;
            new_set_map.insert(hex::encode(field_id), new_value);
        }
        let new_condition = self.condition.rewrite_columns(field_name_id_map)?;
        self.set_map = new_set_map;
        self.condition = new_condition;
        Ok(self)
    }
}

pub(crate) async fn process_update_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    update: TableUpdate,
    mut matched_data_file_rows: HashMap<Uuid, RecordBatchStream>,
) -> ILResult<usize> {
    let inline_update_count = update_inline_rows(tx_helper, table, &update).await?;

    // TODO this could be optimized into update_inline_rows function
    if inline_update_count != 0 {
        rebuild_inline_indexes(
            tx_helper,
            &table.table_id,
            &table.table_schema,
            &table.index_manager,
        )
        .await?;
    }

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;

    let mut file_update_count = 0;
    for data_file_record in data_file_records {
        let update_count =
            if let Some(stream) = matched_data_file_rows.get_mut(&data_file_record.data_file_id) {
                update_data_file_rows_by_matched_rows(
                    tx_helper,
                    table,
                    &update.set_map,
                    stream,
                    data_file_record,
                )
                .await?
            } else {
                update_data_file_rows_by_condition(
                    tx_helper,
                    table,
                    &update.set_map,
                    &update.condition,
                    data_file_record,
                )
                .await?
            };
        file_update_count += update_count;
    }

    Ok(inline_update_count + file_update_count)
}

pub(crate) async fn update_inline_rows(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    update: &TableUpdate,
) -> ILResult<usize> {
    if tx_helper
        .catalog
        .supports_filter(&update.condition, &table.table_schema.arrow_schema)?
    {
        tx_helper
            .update_inline_rows(&table.table_id, &update.set_map, &update.condition)
            .await
    } else {
        let catalog_schema = Arc::new(CatalogSchema::from_arrow(&table.table_schema.arrow_schema)?);
        let row_stream = tx_helper
            .scan_inline_rows(&table.table_id, &catalog_schema, &[], None, None)
            .await?;
        let mut chunk_stream = row_stream.chunks(100);
        let mut updated_row_ids = Vec::new();
        let mut updated_batches = Vec::new();
        while let Some(row_chunk) = chunk_stream.next().await {
            let rows = row_chunk.into_iter().collect::<ILResult<Vec<_>>>()?;
            let record_batch = rows_to_record_batch(&table.table_schema.arrow_schema, &rows)?;
            let bool_array = update.condition.condition_eval(&record_batch)?;
            for (i, v) in bool_array.iter().enumerate() {
                if let Some(v) = v
                    && v
                {
                    updated_row_ids.push(rows[i].uuid(0)?.expect("row_id is not null"));
                }
            }
            let filtered_batch = arrow::compute::filter_record_batch(&record_batch, &bool_array)?;
            let updated_batch = update_record_batch(&filtered_batch, &update.set_map)?;
            updated_batches.push(updated_batch);
        }
        drop(chunk_stream);
        tx_helper
            .delete_inline_rows(&table.table_id, &[], Some(&updated_row_ids))
            .await?;
        process_insert_into_inline_rows(tx_helper, table, &updated_batches).await?;
        Ok(updated_row_ids.len())
    }
}

pub(crate) async fn update_data_file_rows_by_matched_rows(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    set_map: &HashMap<String, Expr>,
    matched_data_file_rows: &mut RecordBatchStream,
    data_file_record: DataFileRecord,
) -> ILResult<usize> {
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
    // TODO we count emit this after parquet reader supports row position
    let row_id_array = read_row_id_array_from_data_file(
        table.storage.as_ref(),
        &data_file_record.relative_path,
        data_file_record.format,
    )
    .await?;
    let row_ids = fixed_size_binary_array_to_uuids(&row_id_array)?;
    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, &row_ids, &updated_row_ids)
        .await?;
    Ok(updated_row_ids.len())
}

pub(crate) async fn update_data_file_rows_by_condition(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    set_map: &HashMap<String, Expr>,
    condition: &Expr,
    data_file_record: DataFileRecord,
) -> ILResult<usize> {
    let mut stream = read_data_file_by_record(
        table.storage.as_ref(),
        &table.table_schema,
        &data_file_record,
        None,
        vec![condition.clone()],
        None,
        1024,
    )
    .await?;

    let mut updated_row_ids = HashSet::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;

        let row_ids = extract_row_ids_from_record_batch(&batch)?;
        updated_row_ids.extend(row_ids);
        let updated_batch = update_record_batch(&batch, set_map)?;
        process_insert_into_inline_rows(tx_helper, table, &[updated_batch]).await?;
    }

    let row_id_array = read_row_id_array_from_data_file(
        table.storage.as_ref(),
        &data_file_record.relative_path,
        data_file_record.format,
    )
    .await?;
    let row_ids = fixed_size_binary_array_to_uuids(&row_id_array)?;

    tx_helper
        .update_data_file_rows_as_invalid(data_file_record, &row_ids, &updated_row_ids)
        .await?;
    Ok(updated_row_ids.len())
}

pub(crate) async fn parallel_find_matched_data_file_rows(
    storage: Arc<dyn Storage>,
    table_schema: TableSchemaRef,
    condition: Expr,
    data_file_records: Vec<DataFileRecord>,
) -> ILResult<HashMap<Uuid, RecordBatchStream>> {
    let mut handles = Vec::new();
    for data_file_record in data_file_records {
        let storage = storage.clone();
        let table_schema = table_schema.clone();
        let condition = condition.clone();
        let handle: JoinHandle<ILResult<(Uuid, RecordBatchStream)>> = tokio::spawn(async move {
            let mut stream = read_data_file_by_record(
                storage.as_ref(),
                &table_schema,
                &data_file_record,
                None,
                vec![condition],
                None,
                1024,
            )
            .await?;

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
