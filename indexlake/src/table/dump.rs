use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use log::{debug, error};
use uuid::Uuid;

use crate::catalog::{
    Catalog, CatalogHelper, CatalogSchema, DataFileRecord, INTERNAL_ROW_ID_FIELD_NAME,
    IndexFileRecord, InlineIndexRecord, RowStream, RowValidity, TransactionHelper,
    rows_to_record_batch,
};
use crate::expr::col;
use crate::index::{IndexBuilder, IndexManager};
use crate::storage::{DataFileFormat, Storage, build_parquet_writer};
use crate::table::{Table, TableConfig, TableSchemaRef, insert_task};
use crate::{ILError, ILResult};

pub(crate) async fn try_run_dump_task(table: &Table) -> ILResult<()> {
    let namespace_id = table.namespace_id;
    let table_id = table.table_id;
    let table_schema = table.table_schema.clone();
    let index_manager = table.index_manager.clone();
    let table_config = table.config.clone();
    let catalog = table.catalog.clone();
    let storage = table.storage.clone();
    tokio::spawn(async move {
        let try_dump_fn = async || {
            let catalog_helper = CatalogHelper::new(catalog.clone());

            let task_id = format!("dump-table-{}", table_id);
            if catalog_helper.task_exists(&task_id).await? {
                return Ok(false);
            }
            
            let inline_row_count = catalog_helper.count_inline_rows(&table_id, &[]).await?;
            if inline_row_count < table_config.inline_row_count_limit as i64 {
                return Ok(false);
            }

            let dump_task = DumpTask {
                task_id,
                namespace_id,
                table_id,
                table_schema: table_schema.clone(),
                index_manager: index_manager.clone(),
                table_config: table_config.clone(),
                catalog: catalog.clone(),
                storage: storage.clone(),
            };

            let continue_dump = dump_task.run().await?;

            Ok::<_, ILError>(continue_dump)
        };

        loop {
            match try_dump_fn().await {
                Ok(continue_dump) => {
                    if !continue_dump {
                        return;
                    }
                }
                Err(e) => {
                    error!("[indexlake] failed to run dump task: {e:?}");
                    return;
                }
            }
        }
    });
    Ok(())
}

pub(crate) struct DumpTask {
    task_id: String,
    namespace_id: Uuid,
    table_id: Uuid,
    table_schema: TableSchemaRef,
    index_manager: Arc<IndexManager>,
    table_config: Arc<TableConfig>,
    catalog: Arc<dyn Catalog>,
    storage: Arc<dyn Storage>,
}

impl DumpTask {
    async fn run(&self) -> ILResult<bool> {
        let now = Instant::now();

        if insert_task(
            &self.catalog,
            self.task_id.clone(),
            Duration::from_secs(12 * 60 * 60),
        )
        .await
        .is_err()
        {
            debug!(
                "[indexlake] Table {} already has a dump task",
                self.table_id
            );
            return Ok(false);
        }

        let mut tx_helper = TransactionHelper::new(&self.catalog).await?;
        let inline_row_count = tx_helper.count_inline_rows(&self.table_id).await?;
        if inline_row_count < self.table_config.inline_row_count_limit as i64 {
            return Ok(false);
        }

        let catalog_schema = Arc::new(CatalogSchema::from_arrow(&self.table_schema.arrow_schema)?);
        let row_stream = tx_helper
            .scan_inline_rows(
                &self.table_id,
                &catalog_schema,
                &[],
                Some(self.table_config.inline_row_count_limit),
                Some(col(INTERNAL_ROW_ID_FIELD_NAME)),
            )
            .await?;

        let data_file_id = uuid::Uuid::now_v7();
        let relative_path = DataFileRecord::build_relative_path(
            &self.namespace_id,
            &self.table_id,
            &data_file_id,
            self.table_config.preferred_data_file_format,
        );

        let mut index_builders = self.index_manager.new_index_builders()?;

        let row_ids = match self.table_config.preferred_data_file_format {
            DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
                self.write_parquet_file(row_stream, &relative_path, &mut index_builders)
                    .await?
            }
        };
        let size = self
            .storage
            .open(&relative_path)
            .await?
            .metadata()
            .await?
            .size;

        if row_ids.len() != self.table_config.inline_row_count_limit {
            self.storage.delete(&relative_path).await?;
            return Err(ILError::internal(format!(
                "Read row count mismatch: {} rows read, expected {}",
                row_ids.len(),
                self.table_config.inline_row_count_limit
            )));
        }

        let mut index_file_records = Vec::new();
        for index_builder in index_builders.iter_mut() {
            let index_file_id = uuid::Uuid::now_v7();
            let relative_path = IndexFileRecord::build_relative_path(
                &self.namespace_id,
                &self.table_id,
                &index_file_id,
            );
            let output_file = self.storage.create(&relative_path).await?;
            index_builder.write_file(output_file).await?;
            let size = self
                .storage
                .open(&relative_path)
                .await?
                .metadata()
                .await?
                .size;
            index_file_records.push(IndexFileRecord {
                index_file_id,
                table_id: self.table_id,
                index_id: index_builder.index_def().index_id,
                data_file_id,
                relative_path,
                size: size as i64,
            });
        }

        tx_helper
            .insert_data_files(&[DataFileRecord {
                data_file_id,
                table_id: self.table_id,
                format: self.table_config.preferred_data_file_format,
                relative_path: relative_path.clone(),
                size: size as i64,
                record_count: row_ids.len() as i64,
                valid_record_count: row_ids.len() as i64,
                validity: RowValidity::new(row_ids.len()),
            }])
            .await?;

        tx_helper.insert_index_files(&index_file_records).await?;

        let deleted_count = tx_helper
            .delete_inline_rows(&self.table_id, &[], Some(&row_ids))
            .await?;
        if deleted_count != row_ids.len() {
            return Err(ILError::internal(format!(
                "Delete row count mismatch: {} inline rows deleted, expected {}",
                deleted_count,
                row_ids.len()
            )));
        }

        rebuild_inline_indexes(
            &mut tx_helper,
            &self.table_id,
            &self.table_schema,
            &self.index_manager,
        )
        .await?;

        tx_helper.delete_task(&self.task_id).await?;

        tx_helper.commit().await?;

        debug!(
            "[indexlake] dumped table {} {} inline rows in {} ms",
            self.table_id,
            self.table_config.inline_row_count_limit,
            now.elapsed().as_millis()
        );

        Ok(true)
    }

    async fn write_parquet_file(
        &self,
        row_stream: RowStream<'_>,
        relative_path: &str,
        index_builders: &mut Vec<Box<dyn IndexBuilder>>,
    ) -> ILResult<Vec<Uuid>> {
        let mut row_ids = Vec::new();

        let output_file = self.storage.create(relative_path).await?;
        let mut arrow_writer = build_parquet_writer(
            output_file,
            self.table_schema.arrow_schema.clone(),
            self.table_config.parquet_row_group_size,
            self.table_config.preferred_data_file_format,
        )?;

        let mut chunk_stream = row_stream.chunks(self.table_config.parquet_row_group_size);

        while let Some(row_chunk) = chunk_stream.next().await {
            let mut rows = Vec::with_capacity(row_chunk.len());
            for row in row_chunk.into_iter() {
                let row = row?;
                let row_id = row.get_row_id()?.expect("row_id is not null");
                row_ids.push(row_id);
                rows.push(row);
            }
            let record_batch = rows_to_record_batch(&self.table_schema.arrow_schema, &rows)?;

            for index_builder in index_builders.iter_mut() {
                index_builder.append(&record_batch)?;
            }

            arrow_writer.write(&record_batch).await?;
        }

        arrow_writer.close().await?;

        Ok(row_ids)
    }
}

pub(crate) async fn rebuild_inline_indexes(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    table_schema: &TableSchemaRef,
    index_manager: &IndexManager,
) -> ILResult<()> {
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&table_schema.arrow_schema)?);
    let row_stream = tx_helper
        .scan_inline_rows(table_id, &catalog_schema, &[], None, None)
        .await?;
    let mut chunk_stream = row_stream.chunks(100);

    let mut inline_index_records = Vec::new();

    let mut index_builders = index_manager.new_index_builders()?;
    let mut counter = 0;
    while let Some(row_chunk) = chunk_stream.next().await {
        let rows = row_chunk.into_iter().collect::<ILResult<Vec<_>>>()?;
        counter += rows.len();
        let record_batch = rows_to_record_batch(&table_schema.arrow_schema, &rows)?;
        for index_builder in index_builders.iter_mut() {
            index_builder.append(&record_batch)?;
        }

        if counter >= 1000 {
            for index_builder in index_builders.iter_mut() {
                let mut index_data = Vec::new();
                index_builder.write_bytes(&mut index_data)?;
                inline_index_records.push(InlineIndexRecord {
                    index_id: index_builder.index_def().index_id,
                    index_data,
                });
            }
            counter = 0;
            index_builders = index_manager.new_index_builders()?;
        }
    }
    drop(chunk_stream);

    // build inline index records for left rows
    if counter > 0 {
        for index_builder in index_builders.iter_mut() {
            let mut index_data = Vec::new();
            index_builder.write_bytes(&mut index_data)?;
            inline_index_records.push(InlineIndexRecord {
                index_id: index_builder.index_def().index_id,
                index_data,
            });
        }
    }

    // delete old inline index records
    tx_helper
        .delete_inline_indexes(&index_manager.index_ids())
        .await?;

    // insert inline index records
    tx_helper
        .insert_inline_indexes(&inline_index_records)
        .await?;

    Ok(())
}
