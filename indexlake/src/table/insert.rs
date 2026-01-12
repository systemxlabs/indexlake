use std::collections::HashMap;

use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit, i256};
use derive_with::With;
use futures::StreamExt;
use uuid::Uuid;

use crate::catalog::{
    Catalog, DataFileRecord, IndexFileRecord, InlineIndexRecord, RowValidity, TransactionHelper,
};
use crate::index::IndexBuilder;
use crate::storage::{DataFileFormat, build_parquet_writer};
use crate::table::Table;
use crate::utils::{rewrite_batch_schema, serialize_array};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, With)]
pub struct TableInsertion {
    pub data: Vec<RecordBatch>,
    pub force_inline: bool,
    pub ignore_row_id: bool,
    pub try_dump: bool,
}

impl TableInsertion {
    pub fn new(data: Vec<RecordBatch>) -> Self {
        Self {
            data,
            force_inline: false,
            ignore_row_id: true,
            try_dump: true,
        }
    }

    pub fn rewrite_columns(mut self, field_name_id_map: &HashMap<String, Uuid>) -> ILResult<Self> {
        let mut new_data = Vec::with_capacity(self.data.len());
        for batch in self.data {
            let new_batch = rewrite_batch_schema(&batch, field_name_id_map)?;
            new_data.push(new_batch);
        }
        self.data = new_data;
        Ok(self)
    }
}

pub(crate) async fn process_insert_into_inline_rows_without_tx(
    table: &Table,
    batches: &[RecordBatch],
) -> ILResult<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let index_builders = table.index_manager.new_index_builders()?;
    let inline_index_records = build_inline_indexes(batches, index_builders)?;

    // insert inline rows
    let sql_values = build_sql_values(batches, table.catalog.as_ref())?;
    let inline_field_names = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();

    let mut tx_helper = table.transaction_helper().await?;
    tx_helper
        .insert_inline_rows(&table.table_id, &inline_field_names, sql_values)
        .await?;

    // insert inline index records
    tx_helper
        .insert_inline_indexes(&inline_index_records)
        .await?;

    tx_helper.commit().await?;

    Ok(())
}

pub(crate) async fn process_insert_into_inline_rows_with_tx(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    batches: &[RecordBatch],
) -> ILResult<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let index_builders = table.index_manager.new_index_builders()?;
    let inline_index_records = build_inline_indexes(batches, index_builders)?;

    // insert inline rows
    let sql_values = build_sql_values(batches, table.catalog.as_ref())?;
    let inline_field_names = batches[0]
        .schema()
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    tx_helper
        .insert_inline_rows(&table.table_id, &inline_field_names, sql_values)
        .await?;

    // insert inline index records
    tx_helper
        .insert_inline_indexes(&inline_index_records)
        .await?;

    Ok(())
}

pub(crate) async fn process_bypass_insert(
    table: &Table,
    batch_stream: RecordBatchStream,
) -> ILResult<usize> {
    let (data_file_records, index_file_records) = match table.config.preferred_data_file_format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            write_parquet_files(table, batch_stream).await?
        }
    };

    let record_count: usize = data_file_records
        .iter()
        .map(|r| r.record_count as usize)
        .sum();

    if record_count > 0 {
        let mut tx_helper = table.transaction_helper().await?;
        tx_helper.insert_data_files(&data_file_records).await?;
        tx_helper.insert_index_files(&index_file_records).await?;
        tx_helper.commit().await?;
    }

    Ok(record_count)
}

pub(crate) fn build_sql_values(
    batches: &[RecordBatch],
    catalog: &dyn Catalog,
) -> ILResult<Vec<Vec<String>>> {
    let num_rows = batches.iter().map(|batch| batch.num_rows()).sum();
    let mut all_sql_values = Vec::with_capacity(num_rows);
    for batch in batches {
        let sql_values = record_batch_to_sql_values(batch, catalog)?;
        all_sql_values.extend(sql_values);
    }
    Ok(all_sql_values)
}

pub(crate) fn build_inline_indexes(
    batches: &[RecordBatch],
    mut index_builders: Vec<Box<dyn IndexBuilder>>,
) -> ILResult<Vec<InlineIndexRecord>> {
    for batch in batches {
        for builder in index_builders.iter_mut() {
            builder.append(batch)?;
        }
    }

    let mut inline_index_records = Vec::new();
    for builder in index_builders.iter_mut() {
        let mut index_data = Vec::new();
        builder.write_bytes(&mut index_data)?;
        inline_index_records.push(InlineIndexRecord {
            index_id: builder.index_def().index_id,
            index_data,
        });
    }
    Ok(inline_index_records)
}

async fn write_parquet_files(
    table: &Table,
    mut batch_stream: RecordBatchStream,
) -> ILResult<(Vec<DataFileRecord>, Vec<IndexFileRecord>)> {
    let row_limit = table.config.inline_row_count_limit;
    let mut data_file_records = Vec::new();
    let mut index_file_records = Vec::new();

    // Current file state
    let mut current_data_file_id: Option<Uuid> = None;
    let mut current_relative_path: Option<String> = None;
    let mut current_writer: Option<
        parquet::arrow::async_writer::AsyncArrowWriter<Box<dyn crate::storage::OutputFile>>,
    > = None;
    let mut current_index_builders: Option<Vec<Box<dyn IndexBuilder>>> = None;
    let mut current_row_count = 0usize;

    while let Some(batch) = batch_stream.next().await {
        let batch = batch?;
        let batch_rows = batch.num_rows();
        if batch_rows == 0 {
            continue;
        }

        let mut batch_offset = 0;
        while batch_offset < batch_rows {
            // Initialize a new file writer if needed
            if current_writer.is_none() {
                let data_file_id = Uuid::now_v7();
                let relative_path = DataFileRecord::build_relative_path(
                    &table.namespace_id,
                    &table.table_id,
                    &data_file_id,
                    table.config.preferred_data_file_format,
                );
                let output_file = table.storage.create(&relative_path).await?;
                let arrow_writer = build_parquet_writer(
                    output_file,
                    table.table_schema.arrow_schema.clone(),
                    table.config.parquet_row_group_size,
                    table.config.preferred_data_file_format,
                )?;

                current_data_file_id = Some(data_file_id);
                current_relative_path = Some(relative_path);
                current_writer = Some(arrow_writer);
                current_index_builders = Some(table.index_manager.new_index_builders()?);
                current_row_count = 0;
            }

            let writer = current_writer.as_mut().unwrap();
            let index_builders = current_index_builders.as_mut().unwrap();

            let remaining_capacity = row_limit.saturating_sub(current_row_count);
            let rows_to_write = (batch_rows - batch_offset).min(remaining_capacity);

            if rows_to_write > 0 {
                let slice = batch.slice(batch_offset, rows_to_write);
                writer.write(&slice).await?;
                for builder in index_builders.iter_mut() {
                    builder.append(&slice)?;
                }
                current_row_count += rows_to_write;
                batch_offset += rows_to_write;
            }

            // Close file if it reached the limit
            if current_row_count >= row_limit {
                let (data_record, idx_records) = finish_parquet_file(
                    table,
                    current_writer.take().unwrap(),
                    current_index_builders.take().unwrap(),
                    current_data_file_id.take().unwrap(),
                    current_relative_path.take().unwrap(),
                    current_row_count,
                )
                .await?;
                data_file_records.push(data_record);
                index_file_records.extend(idx_records);
            }
        }
    }

    // Finish the last file if it has any data
    if let Some(writer) = current_writer
        && current_row_count > 0
    {
        let (data_record, idx_records) = finish_parquet_file(
            table,
            writer,
            current_index_builders.unwrap(),
            current_data_file_id.unwrap(),
            current_relative_path.unwrap(),
            current_row_count,
        )
        .await?;
        data_file_records.push(data_record);
        index_file_records.extend(idx_records);
    }

    Ok((data_file_records, index_file_records))
}

async fn finish_parquet_file(
    table: &Table,
    writer: parquet::arrow::async_writer::AsyncArrowWriter<Box<dyn crate::storage::OutputFile>>,
    mut index_builders: Vec<Box<dyn IndexBuilder>>,
    data_file_id: Uuid,
    relative_path: String,
    record_count: usize,
) -> ILResult<(DataFileRecord, Vec<IndexFileRecord>)> {
    writer.close().await?;

    let size = table
        .storage
        .open(&relative_path)
        .await?
        .metadata()
        .await?
        .size;

    let data_file_record = DataFileRecord {
        data_file_id,
        table_id: table.table_id,
        format: table.config.preferred_data_file_format,
        relative_path,
        size: size as i64,
        record_count: record_count as i64,
        validity: RowValidity::new(record_count),
    };

    let mut index_file_records = Vec::new();
    for index_builder in index_builders.iter_mut() {
        let index_file_id = Uuid::now_v7();
        let index_relative_path = IndexFileRecord::build_relative_path(
            &table.namespace_id,
            &table.table_id,
            &index_file_id,
        );
        let output_file = table.storage.create(&index_relative_path).await?;
        index_builder.write_file(output_file).await?;
        let index_size = table
            .storage
            .open(&index_relative_path)
            .await?
            .metadata()
            .await?
            .size;
        index_file_records.push(IndexFileRecord {
            index_file_id,
            table_id: table.table_id,
            index_id: index_builder.index_def().index_id,
            data_file_id,
            relative_path: index_relative_path,
            size: index_size as i64,
        });
    }

    Ok((data_file_record, index_file_records))
}

macro_rules! extract_sql_values {
    ($array:expr, $array_ty:ty, $convert:expr) => {{
        let mut sql_values = Vec::with_capacity($array.len());
        let array = $array.as_any().downcast_ref::<$array_ty>().ok_or_else(|| {
            ILError::internal(format!(
                "Failed to downcast array to {}",
                stringify!($array_ty),
            ))
        })?;
        for v in array.iter() {
            sql_values.push(match v {
                Some(v) => $convert(v)?,
                None => "NULL".to_string(),
            });
        }
        sql_values
    }};
}

pub(crate) fn array_to_sql_literals(
    array: &dyn Array,
    catalog: &dyn Catalog,
) -> ILResult<Vec<String>> {
    let data_type = array.data_type();
    let literals = match data_type {
        DataType::Boolean => {
            extract_sql_values!(array, BooleanArray, |v: bool| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Int8 => {
            extract_sql_values!(array, Int8Array, |v: i8| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Int16 => {
            extract_sql_values!(array, Int16Array, |v: i16| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Int32 => {
            extract_sql_values!(array, Int32Array, |v: i32| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Int64 => {
            extract_sql_values!(array, Int64Array, |v: i64| Ok::<_, ILError>(v.to_string()))
        }
        DataType::UInt8 => {
            extract_sql_values!(array, UInt8Array, |v: u8| Ok::<_, ILError>(v.to_string()))
        }
        DataType::UInt16 => {
            extract_sql_values!(array, UInt16Array, |v: u16| Ok::<_, ILError>(v.to_string()))
        }
        DataType::UInt32 => {
            extract_sql_values!(array, UInt32Array, |v: u32| Ok::<_, ILError>(v.to_string()))
        }
        DataType::UInt64 => {
            extract_sql_values!(array, UInt64Array, |v: u64| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Float32 => {
            extract_sql_values!(array, Float32Array, |v: f32| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Float64 => {
            extract_sql_values!(array, Float64Array, |v: f64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Timestamp(TimeUnit::Second, _) => {
            extract_sql_values!(array, TimestampSecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            extract_sql_values!(array, TimestampMillisecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            extract_sql_values!(array, TimestampMicrosecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            extract_sql_values!(array, TimestampNanosecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Date32 => {
            extract_sql_values!(array, Date32Array, |v: i32| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Date64 => {
            extract_sql_values!(array, Date64Array, |v: i64| Ok::<_, ILError>(v.to_string()))
        }
        DataType::Time32(TimeUnit::Second) => {
            extract_sql_values!(array, Time32SecondArray, |v: i32| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Time32(TimeUnit::Millisecond) => {
            extract_sql_values!(array, Time32MillisecondArray, |v: i32| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Time64(TimeUnit::Microsecond) => {
            extract_sql_values!(array, Time64MicrosecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Time64(TimeUnit::Nanosecond) => {
            extract_sql_values!(array, Time64NanosecondArray, |v: i64| {
                Ok::<_, ILError>(v.to_string())
            })
        }
        DataType::Binary => {
            extract_sql_values!(array, BinaryArray, |v: &[u8]| {
                Ok::<_, ILError>(catalog.sql_binary_literal(v))
            })
        }
        DataType::FixedSizeBinary(size) => {
            if *size == 16 {
                extract_sql_values!(array, FixedSizeBinaryArray, |v: &[u8]| {
                    Ok::<_, ILError>(catalog.sql_uuid_literal(&Uuid::from_slice(v)?))
                })
            } else {
                extract_sql_values!(array, FixedSizeBinaryArray, |v: &[u8]| {
                    Ok::<_, ILError>(catalog.sql_binary_literal(v))
                })
            }
        }
        DataType::LargeBinary => {
            extract_sql_values!(array, LargeBinaryArray, |v: &[u8]| {
                Ok::<_, ILError>(catalog.sql_binary_literal(v))
            })
        }
        DataType::BinaryView => {
            extract_sql_values!(array, BinaryViewArray, |v: &[u8]| {
                Ok::<_, ILError>(catalog.sql_binary_literal(v))
            })
        }
        DataType::Utf8 => {
            extract_sql_values!(array, StringArray, |v: &str| {
                Ok::<_, ILError>(catalog.sql_string_literal(v))
            })
        }
        DataType::LargeUtf8 => {
            extract_sql_values!(array, LargeStringArray, |v: &str| {
                Ok::<_, ILError>(catalog.sql_string_literal(v))
            })
        }
        DataType::Utf8View => {
            extract_sql_values!(array, StringViewArray, |v: &str| {
                Ok::<_, ILError>(catalog.sql_string_literal(v))
            })
        }
        DataType::List(inner_field) => {
            let mut sql_values = Vec::with_capacity(array.len());
            let array = array
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| ILError::internal("Failed to downcast array to ListArray"))?;
            for v in array.iter() {
                sql_values.push(match v {
                    Some(v) => {
                        catalog.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
                    }
                    None => "NULL".to_string(),
                });
            }
            sql_values
        }
        DataType::FixedSizeList(inner_field, _len) => {
            let mut sql_values = Vec::with_capacity(array.len());
            let array = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| {
                    ILError::internal("Failed to downcast array to FixedSizeListArray")
                })?;
            for v in array.iter() {
                sql_values.push(match v {
                    Some(v) => {
                        catalog.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
                    }
                    None => "NULL".to_string(),
                });
            }
            sql_values
        }
        DataType::LargeList(inner_field) => {
            let mut sql_values = Vec::with_capacity(array.len());
            let array = array
                .as_any()
                .downcast_ref::<LargeListArray>()
                .ok_or_else(|| ILError::internal("Failed to downcast array to LargeListArray"))?;
            for v in array.iter() {
                sql_values.push(match v {
                    Some(v) => {
                        catalog.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
                    }
                    None => "NULL".to_string(),
                });
            }
            sql_values
        }
        DataType::Decimal128(..) => {
            extract_sql_values!(array, Decimal128Array, |v: i128| {
                Ok::<_, ILError>(format!("'{v}'"))
            })
        }
        DataType::Decimal256(..) => {
            extract_sql_values!(array, Decimal256Array, |v: i256| {
                Ok::<_, ILError>(format!("'{v}'"))
            })
        }
        _ => {
            return Err(ILError::not_supported(format!(
                "Unsupported array type: {data_type}",
            )));
        }
    };
    Ok(literals)
}

pub(crate) fn record_batch_to_sql_values(
    record: &RecordBatch,
    catalog: &dyn Catalog,
) -> ILResult<Vec<Vec<String>>> {
    let mut column_values_list = Vec::with_capacity(record.num_columns());
    for array in record.columns() {
        let column_values = array_to_sql_literals(array, catalog)?;
        column_values_list.push(column_values);
    }
    Ok(column_values_list)
}
