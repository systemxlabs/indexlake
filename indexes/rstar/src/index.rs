use std::collections::HashSet;

use indexlake::catalog::Scalar;
use indexlake::expr::{Expr, Function};
use indexlake::index::{FilterIndexEntries, Index, SearchIndexEntries, SearchQuery};
use indexlake::{ILError, ILResult};
use rstar::{AABB, RTree, RTreeObject};
use uuid::Uuid;

use crate::{RStarIndexParams, compute_bounds};

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
}

#[async_trait::async_trait]
impl Index for RStarIndex {
    async fn search(&self, _query: &dyn SearchQuery) -> ILResult<SearchIndexEntries> {
        Err(ILError::not_supported(
            "RStar index does not support search",
        ))
    }

    async fn filter(&self, filters: &[Expr]) -> ILResult<FilterIndexEntries> {
        if filters.is_empty() {
            return Ok(FilterIndexEntries {
                row_ids: Vec::new(),
            });
        }

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

        Ok(FilterIndexEntries {
            row_ids: row_ids.into_iter().collect(),
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

    let selection = rtree.locate_in_envelope_intersecting(&aabb);
    let row_ids = selection
        .into_iter()
        .map(|object| object.row_id)
        .collect::<Vec<_>>();
    Ok(row_ids)
}
