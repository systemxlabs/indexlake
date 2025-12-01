use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use arrow::array::{AsArray, BooleanArray, FixedSizeBinaryArray, RecordBatch};
use arrow::datatypes::SchemaRef;
use arrow_schema::{ArrowError, FieldRef, Schema};
use futures::future::BoxFuture;
use futures::{StreamExt, TryStreamExt};
use parquet::arrow::arrow_reader::{ArrowPredicate, ArrowReaderOptions, RowFilter};
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::arrow::async_writer::AsyncFileWriter;
use parquet::arrow::{
    AsyncArrowWriter, ParquetRecordBatchStreamBuilder, ProjectionMask, parquet_to_arrow_schema,
};
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData, ParquetMetaDataReader};
use parquet::file::properties::{WriterProperties, WriterVersion};
use uuid::Uuid;

use crate::catalog::{DataFileRecord, INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use crate::expr::{Expr, merge_filters, visited_columns};
use crate::storage::{DataFileFormat, InputFile, OutputFile, Storage};
use crate::table::TableSchemaRef;
use crate::utils::{
    build_projection_from_condition, complete_batch_missing_fields,
    extract_row_ids_from_record_batch, project_schema,
};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Clone, Debug)]
pub(crate) struct ExprPredicate {
    filter: Expr,
    projection: ProjectionMask,
    missing_field_values: Vec<(FieldRef, Scalar)>,
}

impl ExprPredicate {
    pub(crate) fn try_new(
        filters: Vec<Expr>,
        projection: ProjectionMask,
        missing_field_values: Vec<(FieldRef, Scalar)>,
    ) -> ILResult<Self> {
        let filter = merge_filters(filters).expect("filters should not be empty");
        Ok(Self {
            filter,
            projection,
            missing_field_values,
        })
    }
}

impl ArrowPredicate for ExprPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        let batch = if self.missing_field_values.is_empty() {
            batch
        } else {
            let mut new_fields = batch
                .schema_ref()
                .fields
                .iter()
                .cloned()
                .collect::<Vec<_>>();
            let mut new_columns = batch.columns().to_vec();
            for (missing_field, value) in self.missing_field_values.iter() {
                new_fields.push(missing_field.clone());
                // TODO this can be improved if some filters only access missing fields
                let array = value
                    .to_array_of_size(batch.num_rows())
                    .map_err(|e| ArrowError::ExternalError(Box::new(e)))?;
                new_columns.push(array);
            }
            let new_schema = Arc::new(Schema::new(new_fields));
            RecordBatch::try_new(new_schema, new_columns)?
        };

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
    storage: &dyn Storage,
    table_schema: &TableSchemaRef,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    filters: Vec<Expr>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    let input_file = storage.open(&data_file_record.relative_path).await?;
    let mut arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
    let parquet_schema = arrow_reader_builder.parquet_schema();
    let arrow_schema = parquet_to_arrow_schema(parquet_schema, None)?;

    let mut missing_field_id_default_value_map = HashMap::new();
    let mut parquet_projection = Vec::new();
    for index in projection
        .clone()
        .unwrap_or((0..table_schema.arrow_schema.fields.len()).collect::<Vec<_>>())
    {
        let internal_field_name = table_schema.arrow_schema.field(index).name();

        if internal_field_name == INTERNAL_ROW_ID_FIELD_NAME
            || arrow_schema.index_of(internal_field_name).is_ok()
        {
            parquet_projection.push(index);
        } else {
            let Ok(field_id) = Uuid::parse_str(internal_field_name) else {
                return Err(ILError::internal(format!(
                    "Fail to parse internal field name {internal_field_name} to uuid"
                )));
            };
            if let Some(default_value) = table_schema.field_id_default_value_map.get(&field_id) {
                missing_field_id_default_value_map.insert(field_id, default_value.clone());
            } else {
                return Err(ILError::internal(format!(
                    "Data file {} doesn't contain internal field name {internal_field_name} and this field has no default value",
                    data_file_record.data_file_id
                )));
            }
        }
    }
    let projection_mask = ProjectionMask::roots(parquet_schema, parquet_projection);

    let arrow_predicate_opt = if filters.is_empty() {
        None
    } else {
        let visited_columns = filters
            .iter()
            .flat_map(visited_columns)
            .collect::<HashSet<_>>();
        let mut predicate_projection = Vec::new();
        let mut missing_field_values = Vec::new();
        for visited_column in visited_columns {
            if let Ok(index) = arrow_schema.index_of(&visited_column) {
                predicate_projection.push(index);
            } else {
                let Ok(field_id) = Uuid::parse_str(&visited_column) else {
                    return Err(ILError::internal(format!(
                        "Cannot parse column name {visited_column} in filter to uuid"
                    )));
                };
                if let Some(default_value) = table_schema.field_id_default_value_map.get(&field_id)
                {
                    let field = table_schema.arrow_schema.fields
                        [table_schema.arrow_schema.index_of(&visited_column)?]
                    .clone();
                    missing_field_values.push((field, default_value.clone()));
                } else {
                    return Err(ILError::internal(format!(
                        "Parquet file doesn't contain column {visited_column} and this column has no default value"
                    )));
                }
            }
        }
        let predicate_projection_mask = ProjectionMask::roots(parquet_schema, predicate_projection);
        Some(ExprPredicate::try_new(
            filters,
            predicate_projection_mask,
            missing_field_values,
        )?)
    };

    if let Some(arrow_predicate) = arrow_predicate_opt {
        arrow_reader_builder =
            arrow_reader_builder.with_row_filter(RowFilter::new(vec![Box::new(arrow_predicate)]));
    }

    let target_schema = Arc::new(project_schema(
        &table_schema.arrow_schema,
        projection.as_ref(),
    )?);
    let stream = arrow_reader_builder
        .with_row_selection(data_file_record.row_selection())
        .with_projection(projection_mask.clone())
        .with_batch_size(batch_size)
        .build()?
        .map(move |batch| {
            let batch = batch?;
            let completed_batch = complete_batch_missing_fields(
                batch,
                target_schema.clone(),
                &missing_field_id_default_value_map,
            )?;
            Ok::<_, ILError>(completed_batch)
        });

    Ok(Box::pin(stream))
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
    let input_file = storage.open(relative_path).await?;
    let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
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
