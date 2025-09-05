use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::Schema;
use futures::StreamExt;
use lance_core::cache::LanceCache;
use lance_encoding::decoder::{DecoderPlugins, FilterExpression};
use lance_file::v2::reader::{FileReader, FileReaderOptions, ReaderProjection};
use lance_file::v2::writer::{FileWriter, FileWriterOptions};
use lance_file::version::LanceFileVersion;
use lance_io::object_store::{ObjectStore, ObjectStoreParams, ObjectStoreRegistry};
use lance_io::scheduler::{ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use uuid::Uuid;

use crate::catalog::{DataFileRecord, INTERNAL_ROW_ID_FIELD_REF};
use crate::expr::{Expr, merge_filters};
use crate::storage::{DataFileFormat, Storage};
use crate::utils::{
    array_to_uuids, build_projection_from_condition, build_row_id_array,
    extract_row_ids_from_record_batch,
};
use crate::{ILError, ILResult, RecordBatchStream};

// TODO fix local fs
pub(crate) async fn build_lance_writer(
    storage: &Storage,
    relative_path: &str,
    schema: &Schema,
    data_file_format: DataFileFormat,
) -> ILResult<FileWriter> {
    let root_path = storage.root_path()?;

    let registry = Arc::new(ObjectStoreRegistry::default());

    let object_store_params = ObjectStoreParams {
        storage_options: Some(storage.storage_options()?),
        ..Default::default()
    };

    let (object_store, _) =
        ObjectStore::from_uri_and_params(registry, &root_path, &object_store_params).await?;
    let object_writer = object_store.create(&relative_path.into()).await?;

    let version = match data_file_format {
        DataFileFormat::LanceV2_0 => LanceFileVersion::V2_0,
        _ => {
            return Err(ILError::invalid_input(format!(
                "Cannot build lance writer for file format: {data_file_format}",
            )));
        }
    };

    let writer = FileWriter::try_new(
        object_writer,
        schema.try_into()?,
        FileWriterOptions {
            format_version: Some(version),
            ..Default::default()
        },
    )?;

    Ok(writer)
}

pub(crate) async fn read_lance_file_by_record(
    storage: &Storage,
    table_schema: &Schema,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    row_ids: Option<&HashSet<Uuid>>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    let root_path = storage.root_path()?;

    let registry = Arc::new(ObjectStoreRegistry::default());

    let object_store_params = ObjectStoreParams {
        storage_options: Some(storage.storage_options()?),
        ..Default::default()
    };

    let (object_store, _) =
        ObjectStore::from_uri_and_params(registry, &root_path, &object_store_params).await?;

    let store_scheduler = ScanScheduler::new(
        object_store.clone(),
        SchedulerConfig::max_bandwidth(&object_store),
    );

    let scheduler = store_scheduler
        .open_file(
            &data_file_record.relative_path.clone().into(),
            &CachedFileSize::unknown(),
        )
        .await?;

    let reader = FileReader::try_open(
        scheduler,
        None,
        Arc::<DecoderPlugins>::default(),
        &LanceCache::no_cache(),
        FileReaderOptions::default(),
    )
    .await?;

    let ranges = data_file_record
        .row_ranges(row_ids)?
        .into_iter()
        .map(|r| r.start as u64..r.end as u64)
        .collect::<Vec<_>>();

    let projection = match projection {
        Some(projection) => {
            let projected_schema = table_schema.project(&projection)?;
            Some(ReaderProjection {
                schema: Arc::new((&projected_schema).try_into()?),
                column_indices: projection.iter().map(|idx| *idx as u32).collect(),
            })
        }
        None => None,
    };

    let mut task_stream = reader.read_tasks(
        lance_io::ReadBatchParams::Ranges(ranges.into()),
        batch_size as u32,
        projection,
        FilterExpression::no_filter(),
    )?;

    let filter = merge_filters(filters);

    let stream = async_stream::stream! {
        while let Some(task) = task_stream.next().await {
        let batch = task.task.await?;
        if let Some(filter) = &filter {
            let bool_array = filter.condition_eval(&batch)?;
            let indices = bool_array
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    if let Some(v) = v
                        && v
                    {
                        Some(i as u64)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let index_array = UInt64Array::from(indices);
            let filtered_batch = arrow::compute::take_record_batch(&batch, &index_array)?;
            yield Ok(filtered_batch);
        } else {
            yield Ok(batch);
        }
        }
    };

    Ok(Box::pin(stream))
}

pub(crate) async fn read_lance_file_by_record_and_row_id_condition(
    storage: &Storage,
    table_schema: &Schema,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    row_id_condition: &Expr,
) -> ILResult<RecordBatchStream> {
    let valid_row_ids = data_file_record.valid_row_ids();
    let valid_row_ids_array = Arc::new(build_row_id_array(valid_row_ids)?) as ArrayRef;

    let schema = Arc::new(Schema::new(vec![INTERNAL_ROW_ID_FIELD_REF.clone()]));
    let batch = RecordBatch::try_new(schema, vec![valid_row_ids_array.clone()])?;

    let bool_array = row_id_condition.condition_eval(&batch)?;

    let mut indices = Vec::new();
    for (i, v) in bool_array.iter().enumerate() {
        if let Some(v) = v
            && v
        {
            indices.push(i as u64);
        }
    }
    if indices.is_empty() {
        return Ok(Box::pin(futures::stream::empty()));
    }
    let index_array = UInt64Array::from(indices);

    let take_array = arrow::compute::take(valid_row_ids_array.as_ref(), &index_array, None)?;
    let match_row_ids = array_to_uuids(&take_array)?
        .into_iter()
        .collect::<HashSet<_>>();

    let stream = read_lance_file_by_record(
        storage,
        table_schema,
        data_file_record,
        projection,
        vec![],
        Some(&match_row_ids),
        1024,
    )
    .await?;
    Ok(stream)
}

pub(crate) async fn find_matched_row_ids_from_lance_file(
    storage: &Storage,
    table_schema: &Schema,
    condition: &Expr,
    data_file_record: &DataFileRecord,
) -> ILResult<HashSet<Uuid>> {
    let mut projection = build_projection_from_condition(table_schema, condition)?;
    // If the condition does not contain the row id column, add it to the projection
    if !projection.contains(&0) {
        projection.insert(0, 0);
    }

    let mut stream = read_lance_file_by_record(
        storage,
        table_schema,
        data_file_record,
        Some(projection),
        vec![condition.clone()],
        None,
        1024,
    )
    .await?;

    let mut matched_row_ids = HashSet::new();
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        let row_ids = extract_row_ids_from_record_batch(&batch)?;
        matched_row_ids.extend(row_ids);
    }
    Ok(matched_row_ids)
}
