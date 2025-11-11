use std::collections::{BTreeMap, HashSet};

use arrow::array::RecordBatch;
use futures::StreamExt;
use log::debug;
use uuid::Uuid;

use crate::{
    ILError, ILResult, RecordBatchStream,
    catalog::{DataFileRecord, IndexFileRecord, RowValidity, TransactionHelper},
    index::IndexBuilder,
    storage::{EntryMode, build_parquet_writer, read_data_file_by_record},
    table::Table,
    utils::extract_row_ids_from_record_batch,
};

#[derive(Debug, Clone)]
pub enum TableOptimization {
    CleanupOrphanFiles { last_modified_before: i64 },
    MergeDataFiles { valid_row_threshold: usize },
}

pub(crate) async fn process_table_optimization(
    table: &Table,
    optimization: TableOptimization,
) -> ILResult<()> {
    match optimization {
        TableOptimization::CleanupOrphanFiles {
            last_modified_before,
        } => cleanup_orphan_files(table, last_modified_before).await,
        TableOptimization::MergeDataFiles {
            valid_row_threshold,
        } => merge_data_files(table, valid_row_threshold).await,
    }
}

async fn cleanup_orphan_files(table: &Table, last_modified_before: i64) -> ILResult<()> {
    let task_id = format!("cleanup-orphan-files-{}", table.table_id);
    let mut tx_helper = TransactionHelper::new(&table.catalog).await?;
    if tx_helper.insert_task(&task_id).await.is_err() {
        debug!(
            "[indexlake] Table {} already has a cleanup orphan files task",
            table.table_id
        );
        return Ok(());
    }

    let data_files = tx_helper.get_data_files(&table.table_id).await?;
    let index_files = tx_helper.get_table_index_files(&table.table_id).await?;
    let existing_files = data_files
        .iter()
        .map(|f| &f.relative_path)
        .chain(index_files.iter().map(|f| &f.relative_path))
        .collect::<HashSet<_>>();

    let table_dir = format!("{}/{}", table.namespace_id, table.table_id);
    let mut dir_entry_stream = table.storage.list(&table_dir).await?;
    while let Some(entry) = dir_entry_stream.next().await {
        let entry = entry?;
        if matches!(entry.metadata.mode, EntryMode::File)
            && let Some(last_modified) = entry.metadata.last_modified
            && last_modified < last_modified_before
        {
            let relative_path = format!("{}/{}", table_dir, entry.name);
            if !existing_files.contains(&relative_path) {
                table.storage.delete(&relative_path).await?;
            }
        }
    }

    tx_helper.delete_task(&task_id).await?;

    tx_helper.commit().await?;

    Ok(())
}

async fn merge_data_files(table: &Table, valid_row_threshold: usize) -> ILResult<()> {
    let task_id = format!("merge-data-files-{}", table.table_id);
    let mut tx_helper = TransactionHelper::new(&table.catalog).await?;
    if tx_helper.insert_task(&task_id).await.is_err() {
        debug!(
            "[indexlake] Table {} already has a merge data files task",
            table.table_id
        );
        return Ok(());
    }

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;
    let matched_data_files = data_file_records
        .into_iter()
        .filter(|record| record.valid_row_count() < valid_row_threshold)
        .collect::<Vec<_>>();

    let mut merge_file_groups = Vec::with_capacity(matched_data_files.len() / 2);
    let mut file_group = Vec::with_capacity(matched_data_files.len());
    let mut file_group_count = 0;
    for (index, data_file) in matched_data_files.iter().enumerate() {
        file_group.push(index);
        file_group_count += data_file.valid_row_count();
        if file_group_count >= table.config.inline_row_count_limit {
            merge_file_groups.push(file_group.clone());
            file_group.clear();
            file_group_count = 0;
        }
    }

    let mut new_data_files = Vec::with_capacity(merge_file_groups.len());
    let mut index_file_records = Vec::new();
    for group in merge_file_groups.iter() {
        let files = group
            .iter()
            .map(|i| &matched_data_files[*i])
            .collect::<Vec<_>>();

        let mut index_builders = table.index_manager.new_index_builders()?;

        let data_file_id = Uuid::now_v7();
        let relative_path = DataFileRecord::build_relative_path(
            &table.namespace_id,
            &table.table_id,
            &data_file_id,
            table.config.preferred_data_file_format,
        );
        stream_merge_group_files(table, &files, &relative_path, &mut index_builders).await?;

        let record_count: usize = files.iter().map(|f| f.valid_row_count()).sum();
        let size = table
            .storage
            .open(&relative_path)
            .await?
            .metadata()
            .await?
            .size;

        new_data_files.push(DataFileRecord {
            data_file_id,
            table_id: table.table_id,
            format: table.config.preferred_data_file_format,
            relative_path,
            size: size as i64,
            record_count: record_count as i64,
            validity: RowValidity::new(record_count),
        });

        for index_builder in index_builders.iter_mut() {
            let index_file_id = Uuid::now_v7();
            let relative_path = IndexFileRecord::build_relative_path(
                &table.namespace_id,
                &table.table_id,
                &index_file_id,
            );
            let output_file = table.storage.create(&relative_path).await?;
            index_builder.write_file(output_file).await?;
            let size = table
                .storage
                .open(&relative_path)
                .await?
                .metadata()
                .await?
                .size;
            index_file_records.push(IndexFileRecord {
                index_file_id,
                table_id: table.table_id,
                index_id: index_builder.index_def().index_id,
                data_file_id,
                relative_path,
                size: size as i64,
            });
        }
    }

    let delete_data_file_ids = merge_file_groups
        .iter()
        .flatten()
        .map(|i| matched_data_files[*i].data_file_id)
        .collect::<Vec<_>>();
    let delete_count = tx_helper.delete_data_files(&delete_data_file_ids).await?;
    if delete_count != delete_data_file_ids.len() {
        return Err(ILError::internal(format!(
            "Delete data file count mismatched，delete count: {delete_count}, data file ids: {delete_data_file_ids:?}",
        )));
    }

    tx_helper
        .delete_index_files_by_data_file_ids(&delete_data_file_ids)
        .await?;

    tx_helper.insert_data_files(&new_data_files).await?;
    tx_helper.insert_index_files(&index_file_records).await?;

    tx_helper.delete_task(&task_id).await?;

    tx_helper.commit().await?;

    debug!(
        "[indexlake] Successfully merged {} data files to {} for table {}",
        delete_data_file_ids.len(),
        new_data_files.len(),
        table.table_id
    );

    Ok(())
}

async fn stream_merge_group_files(
    table: &Table,
    files: &[&DataFileRecord],
    relative_path: &str,
    index_builders: &mut Vec<Box<dyn IndexBuilder>>,
) -> ILResult<()> {
    let output_file = table.storage.create(relative_path).await?;
    let mut parquet_writer = build_parquet_writer(
        output_file,
        table.table_schema.arrow_schema.clone(),
        table.config.parquet_row_group_size,
        table.config.preferred_data_file_format,
    )?;

    let mut streams: Vec<RecordBatchStream> = Vec::with_capacity(files.len());
    for file in files {
        let stream = read_data_file_by_record(
            table.storage.as_ref(),
            &table.table_schema,
            file,
            None,
            vec![],
            None,
            1,
        )
        .await?;
        streams.push(stream);
    }

    let mut stream_map: BTreeMap<Uuid, (usize, RecordBatch)> = BTreeMap::new();
    for (index, stream) in streams.iter_mut().enumerate() {
        if let Some(batch) = stream.next().await {
            let batch = batch?;
            let row_ids = extract_row_ids_from_record_batch(&batch)?;
            debug_assert_eq!(row_ids.len(), 1);
            stream_map.insert(row_ids[0], (index, batch));
        }
    }

    loop {
        if let Some((_, (stream_index, batch))) = stream_map.pop_first() {
            parquet_writer.write(&batch).await?;
            for builder in index_builders.iter_mut() {
                builder.append(&batch)?;
            }

            if let Some(batch) = streams[stream_index].next().await {
                let batch = batch?;
                let row_ids = extract_row_ids_from_record_batch(&batch)?;
                debug_assert_eq!(row_ids.len(), 1);
                stream_map.insert(row_ids[0], (stream_index, batch));
            }
        }

        if stream_map.is_empty() {
            break;
        }
    }
    Ok(())
}
