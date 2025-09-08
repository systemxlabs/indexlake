use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::{DataType, TimeUnit, i256};
use arrow_schema::SchemaRef;
use futures::StreamExt;
use uuid::Uuid;

use crate::catalog::{
    CatalogDatabase, CatalogSchema, DataFileRecord, IndexFileRecord, InlineIndexRecord,
    TransactionHelper, rows_to_record_batch,
};
use crate::index::IndexBuilder;
use crate::storage::{DataFileFormat, build_parquet_writer};
use crate::table::Table;
use crate::utils::{
    extract_row_id_array_from_record_batch, fixed_size_binary_array_to_uuids, serialize_array,
};
use crate::{ILError, ILResult};

pub struct TableInsertion {
    pub data: Vec<RecordBatch>,
    pub force_inline: bool,
    pub try_dump: bool,
}

impl TableInsertion {
    pub fn new(data: Vec<RecordBatch>) -> Self {
        Self {
            data,
            force_inline: false,
            try_dump: true,
        }
    }

    pub fn with_force_inline(mut self, force_inline: bool) -> Self {
        self.force_inline = force_inline;
        self
    }

    pub fn with_try_dump(mut self, try_dump: bool) -> Self {
        self.try_dump = try_dump;
        self
    }
}

pub(crate) async fn process_insert_into_inline_rows(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    batches: &[RecordBatch],
) -> ILResult<()> {
    if batches.is_empty() {
        return Ok(());
    }

    let mut mergeable_index_builders = table.index_manager.new_mergeable_index_builders()?;
    let mut non_mergeable_index_builders =
        table.index_manager.new_non_mergeable_index_builders()?;

    // insert inline rows
    for batch in batches {
        let sql_values = record_batch_to_sql_values(batch, tx_helper.database)?;

        let inline_field_names = batch
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect::<Vec<_>>();

        tx_helper
            .insert_inline_rows(&table.table_id, &inline_field_names, sql_values)
            .await?;
    }

    // append index builders
    for batch in batches {
        for builder in mergeable_index_builders.iter_mut() {
            builder.append(batch)?;
        }
    }

    append_non_mergeable_index_builders(
        tx_helper,
        &table.table_id,
        &table.schema,
        &mut non_mergeable_index_builders,
    )
    .await?;

    // build inline index records
    let mut inline_index_records = Vec::new();
    for builder in mergeable_index_builders.iter_mut() {
        let mut index_data = Vec::new();
        builder.write_bytes(&mut index_data)?;
        inline_index_records.push(InlineIndexRecord {
            index_id: builder.index_def().index_id,
            index_data,
        });
    }
    for builder in non_mergeable_index_builders.iter_mut() {
        let mut index_data = Vec::new();
        builder.write_bytes(&mut index_data)?;
        inline_index_records.push(InlineIndexRecord {
            index_id: builder.index_def().index_id,
            index_data,
        });
    }

    // clean non-mergeable inline index records
    tx_helper
        .delete_inline_indexes(
            &non_mergeable_index_builders
                .iter()
                .map(|builder| builder.index_def().index_id)
                .collect::<Vec<_>>(),
        )
        .await?;

    // insert inline index records
    tx_helper
        .insert_inline_indexes(&inline_index_records)
        .await?;

    Ok(())
}

pub(crate) async fn process_bypass_insert(
    tx_helper: &mut TransactionHelper,
    table: &Table,
    batches: &[RecordBatch],
) -> ILResult<()> {
    let data_file_id = uuid::Uuid::now_v7();
    let relative_path = DataFileRecord::build_relative_path(
        &table.namespace_id,
        &table.table_id,
        &data_file_id,
        table.config.preferred_data_file_format,
    );

    let mut index_builders = table.index_manager.new_index_builders()?;

    let row_ids = match table.config.preferred_data_file_format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            write_parquet_file(table, &relative_path, batches, &mut index_builders).await?
        }
    };
    let record_count = row_ids.len();

    tx_helper
        .insert_data_files(&[DataFileRecord {
            data_file_id,
            table_id: table.table_id,
            format: table.config.preferred_data_file_format,
            relative_path: relative_path.clone(),
            record_count: record_count as i64,
            row_ids,
            validity: vec![true; record_count],
        }])
        .await?;

    // update index files
    let mut index_file_records = Vec::new();
    for index_builder in index_builders.iter_mut() {
        let index_file_id = Uuid::now_v7();
        let relative_path = IndexFileRecord::build_relative_path(
            &table.namespace_id,
            &table.table_id,
            &index_file_id,
        );
        let output_file = table.storage.create_file(&relative_path).await?;
        index_builder.write_file(output_file).await?;
        index_file_records.push(IndexFileRecord {
            index_file_id,
            table_id: table.table_id,
            index_id: index_builder.index_def().index_id,
            data_file_id,
            relative_path,
        });
    }

    tx_helper.insert_index_files(&index_file_records).await?;

    Ok(())
}

async fn write_parquet_file(
    table: &Table,
    relative_path: &str,
    batches: &[RecordBatch],
    index_builders: &mut Vec<Box<dyn IndexBuilder>>,
) -> ILResult<Vec<Uuid>> {
    let output_file = table.storage.create_file(relative_path).await?;

    let mut arrow_writer = build_parquet_writer(
        output_file,
        table.schema.clone(),
        table.config.parquet_row_group_size,
        table.config.preferred_data_file_format,
    )?;

    let mut row_ids = Vec::new();
    for batch in batches {
        let row_id_array = extract_row_id_array_from_record_batch(batch)?;
        row_ids.extend(fixed_size_binary_array_to_uuids(&row_id_array)?);

        arrow_writer.write(batch).await?;
        for builder in index_builders.iter_mut() {
            builder.append(batch)?;
        }
    }

    arrow_writer.close().await?;

    Ok(row_ids)
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
    database: CatalogDatabase,
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
                Ok::<_, ILError>(database.sql_binary_literal(v))
            })
        }
        DataType::FixedSizeBinary(size) => {
            if *size == 16 {
                extract_sql_values!(array, FixedSizeBinaryArray, |v: &[u8]| {
                    Ok::<_, ILError>(database.sql_uuid_literal(&Uuid::from_slice(v)?))
                })
            } else {
                extract_sql_values!(array, FixedSizeBinaryArray, |v: &[u8]| {
                    Ok::<_, ILError>(database.sql_binary_literal(v))
                })
            }
        }
        DataType::LargeBinary => {
            extract_sql_values!(array, LargeBinaryArray, |v: &[u8]| {
                Ok::<_, ILError>(database.sql_binary_literal(v))
            })
        }
        DataType::BinaryView => {
            extract_sql_values!(array, BinaryViewArray, |v: &[u8]| {
                Ok::<_, ILError>(database.sql_binary_literal(v))
            })
        }
        DataType::Utf8 => {
            extract_sql_values!(array, StringArray, |v: &str| {
                Ok::<_, ILError>(database.sql_string_literal(v))
            })
        }
        DataType::LargeUtf8 => {
            extract_sql_values!(array, LargeStringArray, |v: &str| {
                Ok::<_, ILError>(database.sql_string_literal(v))
            })
        }
        DataType::Utf8View => {
            extract_sql_values!(array, StringViewArray, |v: &str| {
                Ok::<_, ILError>(database.sql_string_literal(v))
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
                        database.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
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
                        database.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
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
                        database.sql_binary_literal(&serialize_array(v, inner_field.clone())?)
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
    database: CatalogDatabase,
) -> ILResult<Vec<Vec<String>>> {
    let mut column_values_list = Vec::with_capacity(record.num_columns());
    for array in record.columns() {
        let column_values = array_to_sql_literals(array, database)?;
        column_values_list.push(column_values);
    }
    Ok(column_values_list)
}

pub(crate) async fn append_non_mergeable_index_builders(
    tx_helper: &mut TransactionHelper,
    table_id: &Uuid,
    table_schema: &SchemaRef,
    index_builders: &mut Vec<Box<dyn IndexBuilder>>,
) -> ILResult<()> {
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(table_schema)?);

    let row_stream = tx_helper
        .scan_inline_rows(table_id, &catalog_schema, &[], None)
        .await?;
    let mut inline_stream = row_stream.chunks(100).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let batch = rows_to_record_batch(table_schema, &rows)?;
        Ok::<_, ILError>(batch)
    });

    while let Some(batch) = inline_stream.next().await {
        let batch = batch?;
        for builder in index_builders.iter_mut() {
            builder.append(&batch)?;
        }
    }
    Ok(())
}
