use std::{collections::HashMap, sync::Arc, time::Instant};

use arrow::datatypes::SchemaRef;
use futures::{StreamExt, TryStreamExt};
use log::{debug, error};
use parquet::{arrow::AsyncArrowWriter, file::properties::WriterProperties};

use crate::{
    ILError, ILResult,
    catalog::{
        Catalog, CatalogSchema, DataFileRecord, IndexFileRecord, Row, RowIdMeta, RowStream,
        TransactionHelper, rows_to_record_batch,
    },
    index::{Index, IndexBuilder, IndexDefination, IndexDefinationRef},
    storage::Storage,
    table::{Table, TableConfig},
};

pub(crate) async fn spawn_dump_task(table: &Table) -> ILResult<()> {
    let mut tx_helper = TransactionHelper::new(&table.catalog).await?;
    let dump_row_ids = tx_helper
        .scan_inline_row_ids_with_limit(table.table_id, table.config.inline_row_count_limit)
        .await?;
    tx_helper.commit().await?;
    if dump_row_ids.len() < table.config.inline_row_count_limit {
        debug!(
            "Table {} has less than {} inline rows, skip dump",
            table.table_id, table.config.inline_row_count_limit
        );
        return Ok(());
    }

    let dump_task = DumpTask {
        namespace_id: table.namespace_id,
        table_id: table.table_id,
        table_schema: table.schema.clone(),
        table_indexes: table.indexes.clone(),
        index_kinds: table.index_kinds.clone(),
        table_config: table.config.clone(),
        catalog: table.catalog.clone(),
        storage: table.storage.clone(),
        dump_row_ids,
    };
    tokio::spawn(async move {
        let now = Instant::now();
        if let Err(e) = dump_task.run().await {
            error!("Failed to dump table: {:?}", e);
        }
        debug!(
            "Dump table {} inline rows in {} ms",
            dump_task.table_id,
            now.elapsed().as_millis()
        );
    });
    Ok(())
}

pub(crate) struct DumpTask {
    namespace_id: i64,
    table_id: i64,
    table_schema: SchemaRef,
    table_indexes: HashMap<String, IndexDefinationRef>,
    index_kinds: HashMap<String, Arc<dyn Index>>,
    table_config: Arc<TableConfig>,
    catalog: Arc<dyn Catalog>,
    storage: Arc<Storage>,
    dump_row_ids: Vec<i64>,
}

impl DumpTask {
    async fn run(&self) -> ILResult<()> {
        let mut tx_helper = TransactionHelper::new(&self.catalog).await?;
        if tx_helper.insert_dump_task(self.table_id).await.is_err() {
            debug!("Table {} already has a dump task", self.table_id);
            return Ok(());
        }

        let data_file_id = tx_helper.get_max_data_file_id().await? + 1;

        let catalog_schema = Arc::new(CatalogSchema::from_arrow(&self.table_schema)?);
        let row_stream = tx_helper
            .scan_inline_rows_by_row_ids(self.table_id, &catalog_schema, &self.dump_row_ids)
            .await?;

        let relative_path =
            DataFileRecord::build_relative_path(self.namespace_id, self.table_id, data_file_id);

        let mut index_builders = HashMap::new();
        for (index_name, index_def) in self.table_indexes.iter() {
            let index_kind = self.index_kinds.get(&index_def.kind).ok_or_else(|| {
                ILError::InternalError(format!("Index kind {} not found", index_def.kind))
            })?;
            let index_builder = index_kind.builder(index_def)?;
            index_builders.insert(index_name.clone(), index_builder);
        }

        let (location_map, file_size_bytes, record_count) = self
            .write_dump_file(row_stream, &relative_path, &mut index_builders)
            .await?;

        if record_count != self.dump_row_ids.len() {
            return Err(ILError::InternalError(format!(
                "Read row count mismatch: {} rows read, expected {}",
                record_count,
                self.dump_row_ids.len()
            )));
        }

        tx_helper
            .insert_data_files(&[DataFileRecord {
                data_file_id,
                table_id: self.table_id,
                relative_path,
                file_size_bytes: file_size_bytes as i64,
                record_count: record_count as i64,
                row_id_metas: self
                    .dump_row_ids
                    .iter()
                    .map(|id| RowIdMeta {
                        row_id: *id,
                        valid: true,
                    })
                    .collect(),
            }])
            .await?;

        let mut index_file_id = tx_helper.get_max_index_file_id().await? + 1;
        let mut index_file_records = Vec::new();
        for (index_name, index_builder) in index_builders.iter_mut() {
            let index_def = self
                .table_indexes
                .get(index_name)
                .ok_or_else(|| ILError::InternalError(format!("Index {index_name} not found")))?;
            let relative_path = IndexFileRecord::build_relative_path(
                self.namespace_id,
                self.table_id,
                data_file_id,
                index_def.index_id,
                index_file_id,
            );
            let output_file = self.storage.create_file(&relative_path).await?;
            index_builder.write(output_file).await?;
            index_file_records.push(IndexFileRecord {
                index_file_id,
                index_id: index_def.index_id,
                data_file_id,
                relative_path,
            });
            index_file_id += 1;
        }

        tx_helper.insert_index_files(&index_file_records).await?;

        tx_helper
            .update_row_locations(self.table_id, &location_map)
            .await?;

        let deleted_count = tx_helper
            .delete_inline_rows_by_row_ids(self.table_id, &self.dump_row_ids)
            .await?;
        if deleted_count != self.dump_row_ids.len() {
            return Err(ILError::InternalError(format!(
                "Delete row count mismatch: {} inline rows deleted, expected {}",
                deleted_count,
                self.dump_row_ids.len()
            )));
        }

        tx_helper.delete_dump_task(self.table_id).await?;

        tx_helper.commit().await?;

        Ok(())
    }

    async fn write_dump_file(
        &self,
        row_stream: RowStream<'_>,
        relative_path: &str,
        index_builders: &mut HashMap<String, Box<dyn IndexBuilder>>,
    ) -> ILResult<(HashMap<i64, String>, usize, usize)> {
        let mut location_map = HashMap::new();

        let writer_properties = WriterProperties::builder()
            .set_max_row_group_size(self.table_config.parquet_row_group_size)
            .build();
        let output_file = self.storage.create_file(relative_path).await?;
        let mut arrow_writer = AsyncArrowWriter::try_new(
            output_file,
            self.table_schema.clone(),
            Some(writer_properties),
        )?;

        let mut chunk_stream = row_stream.chunks(self.table_config.parquet_row_group_size);

        let mut row_group_idx = 0;
        let mut record_count = 0;
        while let Some(row_chunk) = chunk_stream.next().await {
            let mut rows = Vec::with_capacity(row_chunk.len());
            for (row_group_offset, row) in row_chunk.into_iter().enumerate() {
                let row = row?;
                let row_id = row.get_row_id()?.expect("row_id is not null");
                location_map.insert(
                    row_id,
                    format!(
                        "parquet:{}:{}:{}",
                        relative_path, row_group_idx, row_group_offset
                    ),
                );
                rows.push(row);
            }
            let record_batch = rows_to_record_batch(&self.table_schema, &rows)?;

            for (_index_name, index_builder) in index_builders.iter_mut() {
                index_builder.update(&record_batch)?;
            }

            arrow_writer.write(&record_batch).await?;

            record_count += record_batch.num_rows();
            row_group_idx += 1;
        }

        let file_size_bytes = arrow_writer.bytes_written();
        arrow_writer.close().await?;

        Ok((location_map, file_size_bytes, record_count))
    }
}
