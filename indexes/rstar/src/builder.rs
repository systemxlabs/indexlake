use std::sync::{Arc, LazyLock};

use arrow::{
    array::{Array, ArrayRef, AsArray, Float64Array, Int64Array, RecordBatch},
    datatypes::{DataType, Field, Float64Type, Int64Type, Schema, SchemaRef},
};
use futures::StreamExt;
use geo::BoundingRect;
use geozero::wkb::FromWkb;
use indexlake::{
    ILError, ILResult,
    index::{Index, IndexBuilder, IndexDefinationRef},
    storage::{InputFile, OutputFile},
    utils::extract_row_id_array_from_record_batch,
};
use parquet::{
    arrow::{AsyncArrowWriter, ParquetRecordBatchStreamBuilder},
    file::properties::WriterProperties,
};
use rstar::{AABB, RTree};

use crate::{IndexTreeObject, RStarIndex, RStarIndexParams, WkbDialect};

pub static RSTAR_INDEX_SCHEMA: LazyLock<SchemaRef> = LazyLock::new(|| {
    Arc::new(Schema::new(vec![
        Field::new("row_id", DataType::Int64, false),
        Field::new("xmin", DataType::Float64, true),
        Field::new("ymin", DataType::Float64, true),
        Field::new("xmax", DataType::Float64, true),
        Field::new("ymax", DataType::Float64, true),
    ]))
});

#[derive(Debug)]
pub struct RStarIndexBuilder {
    index_def: IndexDefinationRef,
    params: RStarIndexParams,
    index_batches: Vec<RecordBatch>,
}

impl RStarIndexBuilder {
    pub fn try_new(index_def: IndexDefinationRef) -> ILResult<Self> {
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
    fn append(&mut self, batch: &RecordBatch) -> ILResult<()> {
        let params = self.index_def.downcast_params::<RStarIndexParams>()?;

        let row_id_array = extract_row_id_array_from_record_batch(&batch)?;

        let key_column_name = &self.index_def.key_columns[0];
        let key_column_index = batch.schema_ref().index_of(&key_column_name)?;
        let key_column = batch.column(key_column_index);
        let aabbs = compute_aabbs(key_column, params.wkb_dialect)?;

        let index_batch = build_index_record_batch(row_id_array, aabbs)?;

        self.index_batches.push(index_batch);

        Ok(())
    }

    async fn read_file(&mut self, input_file: InputFile) -> ILResult<()> {
        let arrow_reader_builder = ParquetRecordBatchStreamBuilder::new(input_file).await?;
        let mut batch_stream = arrow_reader_builder.build()?;

        while let Some(batch) = batch_stream.next().await {
            let batch = batch?;

            self.index_batches.push(batch);
        }

        Ok(())
    }

    async fn write_file(&mut self, output_file: OutputFile) -> ILResult<()> {
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

    fn serialize(&self) -> ILResult<Vec<u8>> {
        todo!()
    }

    fn build(&mut self) -> ILResult<Box<dyn Index>> {
        let num_rows = self
            .index_batches
            .iter()
            .map(|batch| batch.num_rows())
            .sum();
        let mut rtree_objects = Vec::with_capacity(num_rows);
        for batch in self.index_batches.iter() {
            let row_id_array = batch.column(0).as_primitive::<Int64Type>().clone();
            let xmin_array = batch.column(1).as_primitive::<Float64Type>().clone();
            let ymin_array = batch.column(2).as_primitive::<Float64Type>().clone();
            let xmax_array = batch.column(3).as_primitive::<Float64Type>().clone();
            let ymax_array = batch.column(4).as_primitive::<Float64Type>().clone();

            for i in 0..batch.num_rows() {
                if !xmin_array.is_null(i) {
                    let row_id = row_id_array.value(i);
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
        Ok(Box::new(RStarIndex {
            rtree,
            params: self.params.clone(),
        }))
    }
}

fn compute_aabbs(
    array: &ArrayRef,
    wkb_dialect: WkbDialect,
) -> ILResult<Vec<Option<AABB<geo::Coord<f64>>>>> {
    let data_type = array.data_type();
    let mut aabbs = Vec::new();
    match data_type {
        DataType::Binary => {
            let binary_array = array.as_binary::<i32>();
            for wkb_opt in binary_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_aabb(wkb, wkb_dialect)?;
                        aabbs.push(Some(aabb));
                    }
                    None => {
                        aabbs.push(None);
                    }
                }
            }
        }
        DataType::LargeBinary => {
            let large_binary_array = array.as_binary::<i64>();
            for wkb_opt in large_binary_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_aabb(wkb, wkb_dialect)?;
                        aabbs.push(Some(aabb));
                    }
                    None => {
                        aabbs.push(None);
                    }
                }
            }
        }
        DataType::BinaryView => {
            let binary_view_array = array.as_binary_view();
            for wkb_opt in binary_view_array.iter() {
                match wkb_opt {
                    Some(wkb) => {
                        let aabb = compute_aabb(wkb, wkb_dialect)?;
                        aabbs.push(Some(aabb));
                    }
                    None => {
                        aabbs.push(None);
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
    Ok(aabbs)
}

pub(crate) fn compute_aabb(wkb: &[u8], wkb_dialect: WkbDialect) -> ILResult<AABB<geo::Coord<f64>>> {
    let mut rdr = std::io::Cursor::new(wkb);
    let geom = geo::Geometry::from_wkb(&mut rdr, wkb_dialect.to_geozero())
        .map_err(|e| ILError::index(format!("Failed to parse ewkb: {:?}", e)))?;
    if let Some(rect) = geom.bounding_rect() {
        Ok(AABB::from_corners(rect.min(), rect.max()))
    } else {
        Err(ILError::index("Failed to compute AABB of geometry"))
    }
}

fn build_index_record_batch(
    row_id_array: Int64Array,
    aabbs: Vec<Option<AABB<geo::Coord<f64>>>>,
) -> ILResult<RecordBatch> {
    let xmin_array = Float64Array::from(
        aabbs
            .iter()
            .map(|aabb| aabb.map(|aabb| aabb.lower().x))
            .collect::<Vec<_>>(),
    );
    let ymin_array = Float64Array::from(
        aabbs
            .iter()
            .map(|aabb| aabb.map(|aabb| aabb.lower().y))
            .collect::<Vec<_>>(),
    );
    let xmax_array = Float64Array::from(
        aabbs
            .iter()
            .map(|aabb| aabb.map(|aabb| aabb.upper().x))
            .collect::<Vec<_>>(),
    );
    let ymax_array = Float64Array::from(
        aabbs
            .iter()
            .map(|aabb| aabb.map(|aabb| aabb.upper().y))
            .collect::<Vec<_>>(),
    );
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
