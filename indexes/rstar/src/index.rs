use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, BooleanArray, BooleanBuilder, RecordBatch};
use arrow::datatypes::DataType;
use geo::Intersects;
use geozero::ToGeo;
use geozero::wkb::{Ewkb, GpkgWkb, MySQLWkb, SpatiaLiteWkb, Wkb};
use indexlake::catalog::Scalar;
use indexlake::expr::{Expr, Function};
use indexlake::index::{
    FilterIndexEntries, FilterRecheck, Index, RowValidity, SearchIndexEntries, SearchQuery,
};
use indexlake::{ILError, ILResult};
use rstar::{AABB, RTree, RTreeObject};
use uuid::Uuid;

use crate::{RStarIndexParams, WkbDialect, compute_bounds};

#[derive(Debug)]
pub struct IndexTreeObject {
    pub aabb: AABB<[f64; 2]>,
    pub row_id: Uuid,
}

impl RTreeObject for IndexTreeObject {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        self.aabb
    }
}

#[derive(Debug)]
pub struct RStarIndex {
    pub rtree: RTree<IndexTreeObject>,
    pub params: RStarIndexParams,
    pub key_column: String,
    pub row_count: usize,
    pub row_id_to_pos: HashMap<Uuid, usize>,
}

#[async_trait::async_trait]
impl Index for RStarIndex {
    async fn search(
        &self,
        _query: &dyn SearchQuery,
        _dynamic_fields: &[String],
        _validity: &indexlake::index::RowValidity,
        _limit: Option<usize>,
    ) -> ILResult<SearchIndexEntries> {
        Err(ILError::not_supported(
            "RStar index does not support search",
        ))
    }

    async fn filter(
        &self,
        filters: &[Expr],
        validity: &RowValidity,
    ) -> ILResult<FilterIndexEntries> {
        if filters.is_empty() {
            return Ok(FilterIndexEntries {
                row_ids: Vec::new(),
                recheck: None,
            });
        }

        let recheck = Arc::new(IntersectsRecheck::try_new(
            filters,
            &self.key_column,
            &self.params,
        )?);

        let first_filter = &filters[0];
        let mut row_ids = self
            .apply_filter(first_filter)?
            .into_iter()
            .collect::<HashSet<_>>();

        for filter in &filters[1..] {
            let filter_row_ids = self
                .apply_filter(filter)?
                .into_iter()
                .collect::<HashSet<_>>();
            row_ids.retain(|id| filter_row_ids.contains(id));
        }

        // Filter by validity using O(1) HashMap lookup
        row_ids.retain(|row_id| {
            self.row_id_to_pos
                .get(row_id)
                .map(|&pos| validity.is_valid(pos).unwrap_or(false))
                .unwrap_or(false)
        });

        Ok(FilterIndexEntries {
            row_ids: row_ids.into_iter().collect(),
            recheck: Some(recheck),
        })
    }
}

impl RStarIndex {
    fn apply_filter(&self, filter: &Expr) -> ILResult<Vec<Uuid>> {
        match filter {
            Expr::Function(func) => {
                if ["intersects", "st_intersects"]
                    .contains(&func.name.to_ascii_lowercase().as_str())
                {
                    rtree_intersects(&self.rtree, func, &self.params)
                } else {
                    Err(ILError::not_supported(format!(
                        "Not supported filter: {filter}"
                    )))
                }
            }
            _ => Err(ILError::not_supported(format!(
                "Not supported filter: {filter}"
            ))),
        }
    }
}

#[derive(Debug)]
struct IntersectsRecheck {
    key_column: String,
    wkb_dialect: WkbDialect,
    queries: Vec<geo::Geometry<f64>>,
}

impl IntersectsRecheck {
    fn try_new(filters: &[Expr], key_column: &str, params: &RStarIndexParams) -> ILResult<Self> {
        let mut queries = Vec::with_capacity(filters.len());
        for filter in filters {
            let Expr::Function(func) = filter else {
                return Err(ILError::index(format!(
                    "RStar recheck only supports intersects functions, got {filter}"
                )));
            };
            let literal = func
                .args
                .get(1)
                .ok_or_else(|| ILError::index("Intersects function requires two arguments"))?
                .as_literal()?;
            let Scalar::Binary(Some(wkb)) = &literal.value else {
                return Err(ILError::index(
                    "Intersects function must have a literal binary as the second argument",
                ));
            };
            queries.push(wkb_to_geo(wkb, params.wkb_dialect)?);
        }
        Ok(Self {
            key_column: key_column.to_string(),
            wkb_dialect: params.wkb_dialect,
            queries,
        })
    }
}

impl FilterRecheck for IntersectsRecheck {
    fn required_columns(&self) -> Vec<String> {
        vec![self.key_column.clone()]
    }

    fn recheck(&self, batch: &RecordBatch) -> ILResult<BooleanArray> {
        let column_index = batch.schema().index_of(&self.key_column)?;
        let column = batch.column(column_index);
        let mut builder = BooleanBuilder::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            let matched = match wkb_value(column, row)? {
                Some(wkb) => {
                    let geometry = wkb_to_geo(wkb, self.wkb_dialect)?;
                    self.queries.iter().all(|query| query.intersects(&geometry))
                }
                None => false,
            };
            builder.append_value(matched);
        }
        Ok(builder.finish())
    }
}

fn wkb_value(array: &ArrayRef, index: usize) -> ILResult<Option<&[u8]>> {
    if array.is_null(index) {
        return Ok(None);
    }
    match array.data_type() {
        DataType::Binary => Ok(Some(array.as_binary::<i32>().value(index))),
        DataType::LargeBinary => Ok(Some(array.as_binary::<i64>().value(index))),
        DataType::BinaryView => Ok(Some(array.as_binary_view().value(index))),
        data_type => Err(ILError::internal(format!(
            "Unsupported geometry data type: {data_type}"
        ))),
    }
}

fn wkb_to_geo(wkb: &[u8], dialect: WkbDialect) -> ILResult<geo::Geometry<f64>> {
    let result = match dialect {
        WkbDialect::Wkb => Wkb(wkb).to_geo(),
        WkbDialect::Ewkb => Ewkb(wkb).to_geo(),
        WkbDialect::Geopackage => GpkgWkb(wkb).to_geo(),
        WkbDialect::MySQL => MySQLWkb(wkb).to_geo(),
        WkbDialect::SpatiaLite => SpatiaLiteWkb(wkb).to_geo(),
    };
    result.map_err(|e| ILError::index(format!("Failed to decode WKB geometry: {e}")))
}

fn rtree_intersects(
    rtree: &RTree<IndexTreeObject>,
    func: &Function,
    params: &RStarIndexParams,
) -> ILResult<Vec<Uuid>> {
    let literal = func.args[1].as_literal()?;
    let Scalar::Binary(Some(wkb)) = &literal.value else {
        return Err(ILError::internal(
            "Intersects function must have a literal binary as the second argument",
        ));
    };
    let Some(aabb) = compute_bounds(wkb, params.wkb_dialect)? else {
        return Ok(Vec::new());
    };
    let aabb = AABB::from_corners([aabb.min_x(), aabb.min_y()], [aabb.max_x(), aabb.max_y()]);

    let selection = rtree.locate_in_envelope_intersecting(aabb);
    let row_ids = selection
        .into_iter()
        .map(|object| object.row_id)
        .collect::<Vec<_>>();
    Ok(row_ids)
}
