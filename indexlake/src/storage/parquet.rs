use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, UInt64Array};
use arrow::datatypes::{Schema, SchemaRef};
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowReaderOptions, RowFilter};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{
    ArrowSchemaConverter, AsyncArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask,
};
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{WriterProperties, WriterVersion};
use uuid::Uuid;

use crate::catalog::{DataFileRecord, INTERNAL_ROW_ID_FIELD_REF};
use crate::expr::{Expr, ExprPredicate, visited_columns};
use crate::storage::{DataFileFormat, InputFile, OutputFile, Storage};
use crate::utils::{
    array_to_uuids, build_projection_from_condition, build_row_id_array,
    extract_row_ids_from_record_batch,
};
use crate::{ILError, ILResult, RecordBatchStream};

impl AsyncFileReader for InputFile {
    fn get_bytes(
        &mut self,
        range: Range<u64>,
    ) -> BoxFuture<'_, parquet::errors::Result<bytes::Bytes>> {
        Box::pin(async move {
            let buffer = self
                .reader
                .read(range.start..range.end)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))?;
            Ok(buffer.to_bytes())
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
                .with_column_indexes(false)
                .with_page_indexes(false)
                .with_offset_indexes(false);
            let size = self
                .file_size_bytes()
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))?;
            let meta = reader.load_and_finish(self, size).await?;

            Ok(Arc::new(meta))
        })
    }
}

impl AsyncFileWriter for OutputFile {
    fn write(&mut self, bs: bytes::Bytes) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            self.writer
                .write(bs)
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))
        })
    }

    fn complete(&mut self) -> BoxFuture<'_, parquet::errors::Result<()>> {
        Box::pin(async {
            let _ = self
                .writer
                .close()
                .await
                .map_err(|err| parquet::errors::ParquetError::External(Box::new(err)))?;
            Ok(())
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
        .set_max_row_group_size(row_group_size)
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

pub(crate) async fn read_parquet_file_by_record(
    storage: &Storage,
    table_schema: &Schema,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    row_ids: Option<&HashSet<Uuid>>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    let projection_mask = match projection {
        Some(projection) => {
            let parquet_schema = ArrowSchemaConverter::new().convert(table_schema)?;
            ProjectionMask::roots(&parquet_schema, projection)
        }
        None => ProjectionMask::all(),
    };

    let arrow_predicate_opt = if filters.is_empty() {
        None
    } else {
        let visited_columns = filters
            .iter()
            .map(|expr| visited_columns(expr))
            .flatten()
            .collect::<HashSet<_>>();
        let mut predicate_projection = Vec::new();
        for visited_column in visited_columns {
            let index = table_schema.index_of(&visited_column)?;
            predicate_projection.push(index);
        }
        let parquet_schema = ArrowSchemaConverter::new().convert(table_schema)?;
        let predicate_projection_mask =
            ProjectionMask::roots(&parquet_schema, predicate_projection);
        Some(ExprPredicate::try_new(filters, predicate_projection_mask)?)
    };

    let input_file = storage.open_file(&data_file_record.relative_path).await?;
    let mut arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;

    if let Some(arrow_predicate) = &arrow_predicate_opt {
        arrow_reader_builder = arrow_reader_builder
            .with_row_filter(RowFilter::new(vec![Box::new(arrow_predicate.clone())]));
    }

    let stream = arrow_reader_builder
        .with_row_selection(data_file_record.row_selection(row_ids)?)
        .with_projection(projection_mask.clone())
        .with_batch_size(batch_size)
        .build()?
        .map_err(ILError::from);
    Ok(Box::pin(stream))
}

pub(crate) async fn read_parquet_file_by_record_and_row_id_condition(
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

    let stream = read_parquet_file_by_record(
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

pub(crate) async fn find_matched_row_ids_from_parquet_file(
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

    let mut stream = read_parquet_file_by_record(
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
