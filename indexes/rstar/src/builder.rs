use std::sync::{Arc, LazyLock};

use arrow::array::{Array, ArrayRef, AsArray, FixedSizeBinaryArray, Float64Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Float64Type, Schema, SchemaRef};
use futures::StreamExt;
use geozero::GeozeroGeometry;
use geozero::bounds::{Bounds, BoundsProcessor};
use geozero::wkb::{Ewkb, GpkgWkb, MySQLWkb, SpatiaLiteWkb, Wkb};
use indexlake::index::{Index, IndexBuilder, IndexDefinitionRef};
use indexlake::storage::{InputFile, OutputFile};
use indexlake::utils::extract_row_id_array_from_record_batch;
use indexlake::{ILError, ILResult};
use parquet::arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder};
use parquet::file::properties::WriterProperties;
use rstar::{AABB, RTree};
use uuid::Uuid;

use crate::{IndexTreeObject, RStarIndex, RStarIndexParams, WkbDialect};

pub static RSTAR_INDEX_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::FixedSizeBinary(16), false),
        Field::new("xmin", DataType::Float64, true),
        Field::new("ymin", DataType::Float64, true),
        Field::new("xmax", DataType::Float64, true),
        Field::new("ymax", DataType::Float64, true),
    ]))
});

#[derive(Debug)]
pub struct RStarIndexBuilder {
    index_def: IndexDefinitionRef,
    params: RStarIndexParams,
    index_batches: Vec<RecordBatch>,
}

impl RStarIndexBuilder {
    pub fn try_new(index_def: IndexDefinitionRef) -> ILResult<Self> {
        let params = index_def.downcast_params::<RStarIndexParams>()?.clone();
        Ok(Self {
            index_def,
            params,
            index_batches: Vec::new(),
        })
    }
}

#[async_trait::async_trait]
impl IndexBuilder for RStarIndexBuilder {
    fn index_def(&self) -> &IndexDefinitionRef {
        &self.index_def
    }

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let params = self.index_def.downcast_params::<RStarIndexParams>()?;

        let row_id_array = extract_row_id_array_from_record_batch(batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(key_column_name)?;
        let key_column = batch.column(key_column_index);

        let index_batch = build_index_record_batch(key_column, params.wkb_dialect, row_id_array)?;

        self.index_batches.push(index_batch);

        Ok(())
    }

    async fn read_file(&mut self, input_file: Box<dyn InputFile>) -> ILResult<()> {
        let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
        let mut batch_stream = arrow_reader_builder.build()?;

        while let Some(batch) = batch_stream.next().await {
            let batch = batch?;

            self.index_batches.push(batch);
        }

        Ok(())
    }

    async fn write_file(&mut self, output_file: Box<dyn OutputFile>) -> ILResult<()> {
        let writer_properties = WriterProperties::builder()
            .set_max_row_group_size(4096)
            .build();
        let mut arrow_writer = AsyncArrowWriter::try_new(
            output_file,
            RSTAR_INDEX_SCHEMA.clone(),
            Some(writer_properties),
        )?;

        for batch in self.index_batches.iter() {
            arrow_writer.write(batch).await?;
        }

        arrow_writer.close().await?;

        Ok(())
    }

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()> {
        let stream_reader = arrow::ipc::reader::StreamReader::try_new(buf, None)?;
        for batch in stream_reader {
            let batch = batch?;
            self.index_batches.push(batch);
        }
        Ok(())
    }

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()> {
        let mut stream_writer =
            arrow::ipc::writer::StreamWriter::try_new(buf, &RSTAR_INDEX_SCHEMA)?;
        for batch in self.index_batches.iter() {
            stream_writer.write(batch)?;
        }
        stream_writer.finish()?;
        Ok(())
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        Ok(Box::new(self.build_index()?))
    }
}

impl RStarIndexBuilder {
    pub fn build_index(&mut self) -> ILResult<RStarIndex> {
        let num_rows = self
            .index_batches
            .iter()
            .map(|batch| batch.num_rows())
            .sum();
        let mut rtree_objects = Vec::with_capacity(num_rows);
        for batch in self.index_batches.iter() {
            let row_id_array = batch.column(0).as_fixed_size_binary().clone();
            let xmin_array = batch.column(1).as_primitive::<Float64Type>().clone();
            let ymin_array = batch.column(2).as_primitive::<Float64Type>().clone();
            let xmax_array = batch.column(3).as_primitive::<Float64Type>().clone();
            let ymax_array = batch.column(4).as_primitive::<Float64Type>().clone();

            for i in 0..batch.num_rows() {
                if !xmin_array.is_null(i) {
                    let row_id = Uuid::from_slice(row_id_array.value(i))?;
                    let aabb = AABB::from_corners(
                        [xmin_array.value(i), ymin_array.value(i)],
                        [xmax_array.value(i), ymax_array.value(i)],
                    );
                    let object = IndexTreeObject { aabb, row_id };
                    rtree_objects.push(object);
                }
            }
        }
        let rtree = RTree::bulk_load(rtree_objects);
        Ok(RStarIndex {
            rtree,
            params: self.params.clone(),
        })
    }
}

pub(crate) fn compute_bounds(wkb: &[u8], wkb_dialect: WkbDialect) -> ILResult<Option<Bounds>> {
    let mut processor = BoundsProcessor::new();
    match wkb_dialect {
        WkbDialect::Wkb => Wkb(wkb).process_geom(&mut processor),
        WkbDialect::Ewkb => Ewkb(wkb).process_geom(&mut processor),
        WkbDialect::Geopackage => GpkgWkb(wkb).process_geom(&mut processor),
        WkbDialect::MySQL => MySQLWkb(wkb).process_geom(&mut processor),
        WkbDialect::SpatiaLite => SpatiaLiteWkb(wkb).process_geom(&mut processor),
    }
    .map_err(|e| ILError::index(format!("Failed to compute bounds: {e:?}")))
    .map(|_| processor.bounds())
}

fn build_index_record_batch(
    array: &ArrayRef,
    wkb_dialect: WkbDialect,
    row_id_array: FixedSizeBinaryArray,
) -> ILResult<RecordBatch> {
    let data_type = array.data_type();
    let len = array.len();
    let mut xmin_builder = Float64Builder::with_capacity(len);
    let mut ymin_builder = Float64Builder::with_capacity(len);
    let mut xmax_builder = Float64Builder::with_capacity(len);
    let mut ymax_builder = Float64Builder::with_capacity(len);
    match data_type {
        DataType::Binary => {
            let binary_array = array.as_binary::<i32>();
            for wkb_opt in binary_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_bounds(wkb, wkb_dialect)?;
                        if let Some(aabb) = aabb {
                            xmin_builder.append_value(aabb.min_x());
                            ymin_builder.append_value(aabb.min_y());
                            xmax_builder.append_value(aabb.max_x());
                            ymax_builder.append_value(aabb.max_y());
                        } else {
                            xmin_builder.append_null();
                            ymin_builder.append_null();
                            xmax_builder.append_null();
                            ymax_builder.append_null();
                        }
                    }
                    None => {
                        xmin_builder.append_null();
                        ymin_builder.append_null();
                        xmax_builder.append_null();
                        ymax_builder.append_null();
                    }
                }
            }
        }
        DataType::LargeBinary => {
            let large_binary_array = array.as_binary::<i64>();
            for wkb_opt in large_binary_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_bounds(wkb, wkb_dialect)?;
                        if let Some(aabb) = aabb {
                            xmin_builder.append_value(aabb.min_x());
                            ymin_builder.append_value(aabb.min_y());
                            xmax_builder.append_value(aabb.max_x());
                            ymax_builder.append_value(aabb.max_y());
                        } else {
                            xmin_builder.append_null();
                            ymin_builder.append_null();
                            xmax_builder.append_null();
                            ymax_builder.append_null();
                        }
                    }
                    None => {
                        xmin_builder.append_null();
                        ymin_builder.append_null();
                        xmax_builder.append_null();
                        ymax_builder.append_null();
                    }
                }
            }
        }
        DataType::BinaryView => {
            let binary_view_array = array.as_binary_view();
            for wkb_opt in binary_view_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_bounds(wkb, wkb_dialect)?;
                        if let Some(aabb) = aabb {
                            xmin_builder.append_value(aabb.min_x());
                            ymin_builder.append_value(aabb.min_y());
                            xmax_builder.append_value(aabb.max_x());
                            ymax_builder.append_value(aabb.max_y());
                        } else {
                            xmin_builder.append_null();
                            ymin_builder.append_null();
                            xmax_builder.append_null();
                            ymax_builder.append_null();
                        }
                    }
                    None => {
                        xmin_builder.append_null();
                        ymin_builder.append_null();
                        xmax_builder.append_null();
                        ymax_builder.append_null();
                    }
                }
            }
        }
        _ => {
            return Err(ILError::index(format!(
                "Unsupported data type to compute AABB: {data_type}"
            )));
        }
    }

    let xmin_array = xmin_builder.finish();
    let ymin_array = ymin_builder.finish();
    let xmax_array = xmax_builder.finish();
    let ymax_array = ymax_builder.finish();
    let arrays = vec![
        Arc::new(row_id_array) as ArrayRef,
        Arc::new(xmin_array) as ArrayRef,
        Arc::new(ymin_array) as ArrayRef,
        Arc::new(xmax_array) as ArrayRef,
        Arc::new(ymax_array) as ArrayRef,
    ];

    let record_batch = RecordBatch::try_new(RSTAR_INDEX_SCHEMA.clone(), arrays)?;

    Ok(record_batch)
}
