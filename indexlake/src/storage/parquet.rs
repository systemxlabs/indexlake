use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{AsArray, BooleanArray, FixedSizeBinaryArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use arrow_schema::ArrowError;
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{
    ArrowPredicate, ArrowReaderMetadata, ArrowReaderOptions, RowFilter,
};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{
    AsyncArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask, parquet_to_arrow_schema,
};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{WriterProperties, WriterVersion};
use uuid::Uuid;

use crate::catalog::{DataFileRecord, INTERNAL_ROW_ID_FIELD_NAME};
use crate::expr::{Expr, merge_filters, visited_columns};
use crate::storage::prune::{
    PruneOutcome, extract_stats_predicates, prune_file_row_groups, read_footer_metadata,
};
use crate::storage::{DataFileFormat, InputFile, OutputFile, Storage};
use crate::table::TableSchemaRef;
use crate::utils::{build_projection_from_condition, extract_row_ids_from_record_batch};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Clone, Debug)]
pub(crate) struct ExprPredicate {
    filter: Expr,
    projection: ProjectionMask,
}

impl ExprPredicate {
    pub(crate) fn try_new(filters: Vec<Expr>, projection: ProjectionMask) -> ILResult<Self> {
        let filter = merge_filters(filters).expect("filters should not be empty");
        Ok(Self { filter, projection })
    }
}

impl ArrowPredicate for ExprPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        let array = self
            .filter
            .eval(&batch)
            .map_err(|e| ArrowError::from_external_error(Box::new(e)))?
            .into_array(batch.num_rows())
            .map_err(|e| ArrowError::from_external_error(Box::new(e)))?;
        let bool_array = array.as_boolean_opt().ok_or_else(|| {
            ArrowError::ComputeError(format!(
                "ExprPredicate evaluation expected boolean array, got {}",
                array.data_type()
            ))
        })?;

        Ok(bool_array.clone())
    }
}

impl AsyncFileReader for Box<dyn InputFile> {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        Box::pin(async move {
            self.read(range.start..range.end)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<bytes::Bytes>>> {
        Box::pin(async move {
            self.read_ranges(ranges)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }

    // TODO respect options
    fn get_metadata(
        &mut self,
        _options: Option<&'_ ArrowReaderOptions>,
    ) -> BoxFuture<'_, parquet::errors::Result<Arc<ParquetMetaData>>> {
        Box::pin(async {
            let reader = ParquetMetaDataReader::new()
                .with_prefetch_hint(None)
                .with_column_index_policy(PageIndexPolicy::Skip)
                .with_page_index_policy(PageIndexPolicy::Skip)
                .with_offset_index_policy(PageIndexPolicy::Skip);
            let size = self
                .metadata()
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))?
                .size;
            let meta = reader.load_and_finish(self, size).await?;

            Ok(Arc::new(meta))
        })
    }
}

impl AsyncFileWriter for Box<dyn OutputFile> {
    fn write(&mut self, bs: bytes::Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            OutputFile::write(self, bs)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.close()
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }
}

pub(crate) fn build_parquet_writer<W: AsyncFileWriter>(
    writer: W,
    schema: SchemaRef,
    row_group_size: usize,
    data_file_format: DataFileFormat,
) -> ILResult<AsyncArrowWriter<W>> {
    let writer_properties = WriterProperties::builder()
        .set_max_row_group_row_count(Some(row_group_size))
        .set_writer_version(match data_file_format {
            DataFileFormat::ParquetV1 => WriterVersion::PARQUET_1_0,
            DataFileFormat::ParquetV2 => WriterVersion::PARQUET_2_0,
        })
        .build();
    Ok(AsyncArrowWriter::try_new(
        writer,
        schema,
        Some(writer_properties),
    )?)
}

/// Assemble the record batch stream for a single data file. The file's footer
/// is read exactly once and injected into the reader builder, so row-group
/// statistics pruning and decoding share the same metadata without a second
/// metadata read.
///
/// Returns `Ok(None)` when statistics pruning proved that the file cannot
/// contain any matching row (all row groups pruned); the caller should skip
/// the file entirely.
async fn open_parquet_file_stream(
    mut input_file: Box<dyn InputFile>,
    table_schema: &TableSchemaRef,
    record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    batch_size: usize,
) -> ILResult<Option<RecordBatchStream>> {
    // The footer suffix is located from `record.size` as recorded in the
    // catalog (written by fstat at commit time), avoiding a metadata stat per
    // file. A file rewritten out-of-band therefore fails loudly here instead
    // of being read with a misaligned row grid.
    let metadata = Arc::new(
        read_footer_metadata(&mut *input_file, &record.relative_path, record.size as u64).await?,
    );
    let arrow_reader_metadata =
        ArrowReaderMetadata::try_new(metadata.clone(), ArrowReaderOptions::default())?;
    let mut arrow_reader_builder =
        ParquetRecordBatchStreamBuilder::new_with_metadata(input_file, arrow_reader_metadata);
    let parquet_schema = arrow_reader_builder.parquet_schema();
    let arrow_schema = parquet_to_arrow_schema(parquet_schema, None)?;

    let mut parquet_projection = Vec::new();
    for index in
        projection.unwrap_or((0..table_schema.arrow_schema.fields.len()).collect::<Vec<_>>())
    {
        let internal_field_name = table_schema.arrow_schema.field(index).name();

        if internal_field_name == INTERNAL_ROW_ID_FIELD_NAME
            || arrow_schema.index_of(internal_field_name).is_ok()
        {
            parquet_projection.push(index);
        } else {
            return Err(ILError::internal(format!(
                "Data file {} doesn't contain internal field name {internal_field_name}",
                record.data_file_id
            )));
        }
    }
    let projection_mask = ProjectionMask::roots(parquet_schema, parquet_projection);

    // Statistics pruning is judged against the table schema before the
    // filters are consumed by the row filter below.
    let stats_predicates = extract_stats_predicates(&filters, &table_schema.arrow_schema);

    let arrow_predicate_opt = if filters.is_empty() {
        None
    } else {
        let visited_columns = filters
            .iter()
            .flat_map(visited_columns)
            .collect::<HashSet<_>>();
        let mut predicate_projection = Vec::new();
        for visited_column in visited_columns {
            if let Ok(index) = arrow_schema.index_of(&visited_column) {
                predicate_projection.push(index);
            } else {
                return Err(ILError::internal(format!(
                    "Parquet file doesn't contain column {visited_column}"
                )));
            }
        }
        let predicate_projection_mask = ProjectionMask::roots(parquet_schema, predicate_projection);
        Some(ExprPredicate::try_new(filters, predicate_projection_mask)?)
    };

    // Row-group pruning: pure in-memory judgment over the footer already in
    // hand. Intersect the row-group selection with the validity (delete
    // bitmap) selection; `prune_file_row_groups` abandons pruning when the
    // footer row total doesn't match the catalog record count, so the two
    // selections are always aligned on the same row grid here.
    let row_selection = if stats_predicates.is_empty() {
        record.row_selection()
    } else {
        match prune_file_row_groups(&metadata, &stats_predicates, record.record_count as usize) {
            Some(PruneOutcome::Skip) => return Ok(None),
            Some(PruneOutcome::Partial(row_group_selection)) => {
                record.row_selection().intersection(&row_group_selection)
            }
            None => record.row_selection(),
        }
    };

    arrow_reader_builder = arrow_reader_builder.with_projection(projection_mask);
    arrow_reader_builder = arrow_reader_builder.with_row_selection(row_selection);
    if let Some(arrow_predicate) = arrow_predicate_opt {
        arrow_reader_builder =
            arrow_reader_builder.with_row_filter(RowFilter::new(vec![Box::new(arrow_predicate)]));
    }

    let stream = arrow_reader_builder
        .with_batch_size(batch_size)
        .build()?
        .map(|batch| Ok::<_, ILError>(batch?));

    Ok(Some(Box::pin(stream)))
}

pub(crate) async fn read_parquet_file_by_record(
    storage: &dyn Storage,
    table_schema: &TableSchemaRef,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    let input_file = storage.open(&data_file_record.relative_path).await?;
    match open_parquet_file_stream(
        input_file,
        table_schema,
        data_file_record,
        projection,
        filters,
        batch_size,
    )
    .await?
    {
        Some(stream) => Ok(stream),
        None => Ok(Box::pin(futures::stream::empty())),
    }
}

/// Read data files as one ordered record batch stream with bounded
/// cross-file concurrency. Each file's whole assembly (open + footer read +
/// builder construction) runs as one future, so `buffered(concurrency)`
/// overlaps file N+1's assembly with file N's decoding. Output order equals
/// the input record order.
pub(crate) async fn read_data_files_by_record(
    storage: Arc<dyn Storage>,
    table_schema: TableSchemaRef,
    data_file_records: &[DataFileRecord],
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    if data_file_records.is_empty() {
        return Ok(Box::pin(futures::stream::empty()));
    }
    let concurrency = scan_concurrency().min(data_file_records.len());

    // Keep the per-file future's captured data owned: closures capturing
    // references inside this future trip rustc's higher-ranked lifetime
    // inference when the enclosing future is stored in the scan machinery.
    // The stream must own the records: the boxed stream is 'static.
    let records = data_file_records.to_vec();
    let futures = records.into_iter().map(move |record| {
        let storage = storage.clone();
        let table_schema = table_schema.clone();
        let projection = projection.clone();
        let filters = filters.clone();
        async move {
            let input_file = storage.open(&record.relative_path).await?;
            // Format dispatch: only parquet data files exist today
            // (DataFileFormat::ParquetV1/V2); a non-parquet format must not
            // go through open_parquet_file_stream.
            open_parquet_file_stream(
                input_file,
                &table_schema,
                &record,
                projection,
                filters,
                batch_size,
            )
            .await
        }
    });

    let stream = futures::stream::iter(futures)
        .buffered(concurrency)
        // Drop files that were fully pruned; propagate errors.
        .try_filter_map(|stream| async move { Ok(stream) })
        .try_flatten();

    Ok(Box::pin(stream))
}

/// Cross-file scan concurrency: `IL_SCAN_CONCURRENCY` overrides, defaulting
/// to the CPU count clamped to [4, 16] — decoding is CPU-bound so extra
/// concurrency mostly adds memory and scheduling overhead.
fn scan_concurrency() -> usize {
    if let Some(n) = std::env::var("IL_SCAN_CONCURRENCY")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|n| *n > 0)
    {
        return n;
    }
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(4, 16)
}

pub(crate) async fn find_matched_row_ids_from_parquet_file(
    storage: &dyn Storage,
    table_schema: &TableSchemaRef,
    condition: &Expr,
    data_file_record: &DataFileRecord,
) -> ILResult<HashSet<Uuid>> {
    let mut projection = build_projection_from_condition(&table_schema.arrow_schema, condition)?;
    // If the condition does not contain the row id column, add it to the projection
    if !projection.contains(&0) {
        projection.insert(0, 0);
    }

    let mut stream = read_parquet_file_by_record(
        storage,
        table_schema,
        data_file_record,
        Some(projection),
        vec![condition.clone()],
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

pub(crate) async fn read_row_id_array_from_parquet(
    storage: &dyn Storage,
    relative_path: &str,
) -> ILResult<FixedSizeBinaryArray> {
    let mut input_file = storage.open(relative_path).await?;
    let size = input_file.metadata().await?.size;
    let metadata = Arc::new(read_footer_metadata(&mut *input_file, relative_path, size).await?);
    let arrow_reader_metadata =
        ArrowReaderMetadata::try_new(metadata, ArrowReaderOptions::default())?;
    let arrow_reader_builder =
        ParquetRecordBatchStreamBuilder::new_with_metadata(input_file, arrow_reader_metadata);
    let parquet_schema = arrow_reader_builder.parquet_schema();

    let projection_mask = ProjectionMask::roots(parquet_schema, [0]);

    let stream = arrow_reader_builder
        .with_projection(projection_mask)
        .build()?
        .map_err(ILError::from);

    let batches = stream.try_collect::<Vec<_>>().await?;

    let arrays = batches
        .iter()
        .map(|b| b.column(0).as_ref())
        .collect::<Vec<_>>();
    let array = arrow::compute::concat(&arrays)?;

    let array = array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            ILError::internal("Can not downcast row id array to FixedSizeBinaryArray")
        })?;

    Ok(array.clone())
}
