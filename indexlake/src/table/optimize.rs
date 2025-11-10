use std::collections::HashSet;

use futures::StreamExt;
use log::debug;
use uuid::Uuid;

use crate::{
    ILError, ILResult,
    catalog::{DataFileRecord, RowValidity, TransactionHelper},
    table::Table,
};

#[derive(Debug, Clone)]
pub enum TableOptimization {
    CleanupOrphanFiles,
    MergeDataFiles { valid_row_threshold: usize },
}

pub(crate) async fn process_table_optimization(
    table: &Table,
    optimization: TableOptimization,
) -> ILResult<()> {
    match optimization {
        TableOptimization::CleanupOrphanFiles => cleanup_orphan_files(table).await,
        TableOptimization::MergeDataFiles {
            valid_row_threshold,
        } => merge_data_files(table, valid_row_threshold).await,
    }
}

async fn cleanup_orphan_files(table: &Table) -> ILResult<()> {
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
        let relative_path = format!("{}/{}", table_dir, entry.name);
        if !existing_files.contains(&relative_path) {
            table.storage.delete(&relative_path).await?;
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
    for group in merge_file_groups.iter() {
        let files = group
            .iter()
            .map(|i| &matched_data_files[*i])
            .collect::<Vec<_>>();

        let data_file_id = Uuid::now_v7();
        let relative_path = DataFileRecord::build_relative_path(
            &table.namespace_id,
            &table.table_id,
            &data_file_id,
            table.config.preferred_data_file_format,
        );
        stream_merge_group_files(&files, &relative_path).await?;

        let record_count: usize = files.iter().map(|f| f.valid_row_count()).sum();
        new_data_files.push(DataFileRecord {
            data_file_id,
            table_id: table.table_id,
            format: table.config.preferred_data_file_format,
            relative_path,
            record_count: record_count as i64,
            validity: RowValidity::new(record_count),
        });
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

    tx_helper.insert_data_files(&new_data_files).await?;

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
    _files: &[&DataFileRecord],
    _relative_path: &str,
) -> ILResult<()> {
    // TODO stream merge group files
    todo!()
}
