use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow_schema::{FieldRef, Schema};
use futures::StreamExt;
use uuid::Uuid;

use crate::{
    ILError, ILResult,
    catalog::{DataFileRecord, FieldRecord, IndexFileRecord, RowValidity, TransactionHelper},
    check_schema_contains_system_column,
    expr::{Expr, lit},
    storage::{build_parquet_writer, read_data_file_by_record},
    table::{
        Table, TableSchema, TableSchemaRef, eval_default_expr, spawn_storage_data_files_clean_task,
        spawn_storage_index_files_clean_task,
    },
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
        default_value: Expr,
    },
    DropColumn {
        name: String,
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
            Ok(())
        }
        TableAlter::RenameColumn { old_name, new_name } => {
            let Some(field_record) = tx_helper
                .get_table_field(&table.table_id, &old_name)
                .await?
            else {
                return Err(ILError::invalid_input(format!(
                    "Field name {old_name} not found for table id {}",
                    table.table_id
                )));
            };
            tx_helper
                .update_field_name(&field_record.field_id, &new_name)
                .await?;
            Ok(())
        }
        TableAlter::AddColumn {
            field,
            default_value,
        } => alter_add_column(tx_helper, table, field, default_value).await,
        TableAlter::DropColumn { name } => {
            let Some(field_id) = table.table_schema.field_name_id_map.get(&name) else {
                return Ok(());
            };
            if table.index_manager.any_index_contains_field(field_id) {
                return Err(ILError::invalid_input(format!(
                    "Cannot drop column {name} because it is used in an index"
                )));
            }
            tx_helper
                .delete_field_by_name(&table.table_id, &name)
                .await?;
            tx_helper
                .alter_drop_column(&table.table_id, field_id)
                .await?;
            Ok(())
        }
    }
}

pub(crate) async fn alter_add_column(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    field: FieldRef,
    default_value: Expr,
) -> ILResult<()> {
    let schema = Schema::new(vec![field.clone()]);
    check_schema_contains_system_column(&schema)?;

    if tx_helper
        .get_table_field(&table.table_id, field.name())
        .await?
        .is_some()
    {
        return Err(ILError::invalid_input(format!(
            "Field name {} already exists for table id {}",
            field.name(),
            table.table_id
        )));
    }

    let default_value = default_value.rewrite_columns(&table.table_schema.field_name_id_map)?;
    super::check_default_expr(
        field.as_ref(),
        &default_value,
        &table.table_schema.arrow_schema,
    )?;

    let field_id = Uuid::now_v7();
    let field_record = FieldRecord::new(
        field_id,
        table.table_id,
        field.as_ref(),
        Some(default_value.clone()),
    );
    tx_helper.insert_fields(&[field_record]).await?;

    tx_helper
        .alter_add_column(&table.table_id, &field_id, field.data_type())
        .await?;

    let set_map = HashMap::from([(hex::encode(field_id), default_value.clone())]);
    tx_helper
        .update_inline_rows(&table.table_id, &set_map, &lit(true))
        .await?;

    let old_table_schema = table.table_schema.clone();
    let field_records = tx_helper.get_table_fields(&table.table_id).await?;
    let new_table_schema = Arc::new(TableSchema::new(
        &field_records,
        table.table_schema.arrow_schema.metadata().clone(),
    ));

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;
    if data_file_records.is_empty() {
        return Ok(());
    }

    let (new_data_files, new_index_files) = rewrite_data_files_add_column(
        table,
        &old_table_schema,
        &new_table_schema,
        field_id,
        &default_value,
        &data_file_records,
    )
    .await?;

    let old_data_file_ids = data_file_records
        .iter()
        .map(|r| r.data_file_id)
        .collect::<Vec<_>>();
    let old_index_files = tx_helper
        .get_table_index_files(&table.table_id)
        .await?
        .into_iter()
        .filter(|r| old_data_file_ids.contains(&r.data_file_id))
        .collect::<Vec<_>>();

    tx_helper
        .delete_index_files_by_data_file_ids(&old_data_file_ids)
        .await?;
    tx_helper.delete_data_files(&old_data_file_ids).await?;
    tx_helper.insert_data_files(&new_data_files).await?;
    tx_helper.insert_index_files(&new_index_files).await?;

    if !data_file_records.is_empty() {
        spawn_storage_data_files_clean_task(table.storage.clone(), data_file_records);
    }
    if !old_index_files.is_empty() {
        spawn_storage_index_files_clean_task(table.storage.clone(), old_index_files);
    }

    Ok(())
}

async fn rewrite_data_files_add_column(
    table: &Table,
    old_table_schema: &TableSchemaRef,
    new_table_schema: &TableSchemaRef,
    field_id: Uuid,
    default_value: &Expr,
    data_file_records: &[DataFileRecord],
) -> ILResult<(Vec<DataFileRecord>, Vec<IndexFileRecord>)> {
    let new_field_name = hex::encode(field_id);
    let mut new_data_files = Vec::with_capacity(data_file_records.len());
    let mut new_index_files = Vec::new();
    let mut created_paths = Vec::new();

    let result: ILResult<()> = async {
        for data_file_record in data_file_records {
            let data_file_id = Uuid::now_v7();
            let relative_path = DataFileRecord::build_relative_path(
                &table.namespace_id,
                &table.table_id,
                &data_file_id,
                table.config.preferred_data_file_format,
            );
            created_paths.push(relative_path.clone());

            let output_file = table.storage.create(&relative_path).await?;
            let mut writer = build_parquet_writer(
                output_file,
                new_table_schema.arrow_schema.clone(),
                table.config.parquet_row_group_size,
                table.config.preferred_data_file_format,
            )?;
            let mut index_builders = table.index_manager.new_index_builders()?;

            let mut stream = read_data_file_by_record(
                table.storage.as_ref(),
                old_table_schema,
                data_file_record,
                None,
                vec![],
                None,
                1024,
            )
            .await?;
            let mut record_count = 0usize;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                let new_batch = add_default_column_to_batch(
                    &batch,
                    new_table_schema,
                    &new_field_name,
                    default_value,
                )?;
                record_count += new_batch.num_rows();
                writer.write(&new_batch).await?;
                for builder in index_builders.iter_mut() {
                    builder.append(&new_batch)?;
                }
            }
            writer.close().await?;

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

            for builder in index_builders.iter_mut() {
                let index_file_id = Uuid::now_v7();
                let index_relative_path = IndexFileRecord::build_relative_path(
                    &table.namespace_id,
                    &table.table_id,
                    &index_file_id,
                );
                created_paths.push(index_relative_path.clone());
                let output_file = table.storage.create(&index_relative_path).await?;
                builder.write_file(output_file).await?;
                let index_size = table
                    .storage
                    .open(&index_relative_path)
                    .await?
                    .metadata()
                    .await?
                    .size;
                new_index_files.push(IndexFileRecord {
                    index_file_id,
                    table_id: table.table_id,
                    index_id: builder.index_def().index_id,
                    data_file_id,
                    relative_path: index_relative_path,
                    size: index_size as i64,
                });
            }
        }
        Ok(())
    }
    .await;

    if let Err(err) = result {
        for path in created_paths {
            let _ = table.storage.delete(&path).await;
        }
        return Err(err);
    }

    Ok((new_data_files, new_index_files))
}

fn add_default_column_to_batch(
    batch: &RecordBatch,
    new_table_schema: &TableSchemaRef,
    new_field_name: &str,
    default_value: &Expr,
) -> ILResult<RecordBatch> {
    if batch.schema().index_of(new_field_name).is_ok() {
        return Err(ILError::internal(format!(
            "Default column {new_field_name} already exists in batch"
        )));
    }

    let new_field = new_table_schema
        .arrow_schema
        .field_with_name(new_field_name)?;
    let new_array = eval_default_expr(default_value, batch, new_field)?;

    let mut columns_by_name = HashMap::with_capacity(batch.num_columns());
    for (idx, field) in batch.schema().fields().iter().enumerate() {
        columns_by_name.insert(field.name().clone(), batch.column(idx).clone());
    }

    let mut new_columns = Vec::with_capacity(new_table_schema.arrow_schema.fields().len());
    for field in new_table_schema.arrow_schema.fields() {
        if field.name() == new_field_name {
            new_columns.push(new_array.clone());
            continue;
        }
        let column = columns_by_name.get(field.name()).ok_or_else(|| {
            ILError::internal(format!("Column {} not found in batch", field.name()))
        })?;
        new_columns.push(column.clone());
    }

    Ok(RecordBatch::try_new(
        new_table_schema.arrow_schema.clone(),
        new_columns,
    )?)
}
