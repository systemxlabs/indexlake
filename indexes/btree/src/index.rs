use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::ops::Bound::{Excluded, Included, Unbounded};

use indexlake::catalog::Scalar;
use indexlake::expr::{BinaryOp, Expr};
use indexlake::index::{FilterIndexEntries, Index, SearchIndexEntries, SearchQuery};
use indexlake::utils::build_row_id_array;
use indexlake::{ILError, ILResult};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedScalar(pub Scalar);

impl Ord for OrderedScalar {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.0.partial_cmp(&other.0) {
            Some(ord) => ord,
            None => {
                if self.0.is_null() && other.0.is_null() {
                    Ordering::Equal
                } else if self.0.is_null() {
                    Ordering::Less
                } else if other.0.is_null() {
                    Ordering::Greater
                } else {
                    panic!("Can not cmp scalar: {} and {}", self.0, other.0);
                }
            }
        }
    }
}

impl PartialOrd for OrderedScalar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
pub struct BTreeIndex {
    btree: BTreeMap<OrderedScalar, Vec<Uuid>>,
}

impl Default for BTreeIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl BTreeIndex {
    pub fn new() -> Self {
        Self {
            btree: BTreeMap::new(),
        }
    }

    pub fn insert(&mut self, key: OrderedScalar, row_id: Uuid) -> ILResult<()> {
        self.btree.entry(key).or_default().push(row_id);
        Ok(())
    }

    fn point_query(&self, point: &OrderedScalar) -> Vec<Uuid> {
        self.btree.get(point).cloned().unwrap_or_default()
    }

    fn range_query(&self, range: (Bound<&OrderedScalar>, Bound<&OrderedScalar>)) -> Vec<Uuid> {
        let mut results = Vec::new();

        for (_, row_ids) in self.btree.range(range) {
            results.extend_from_slice(row_ids);
        }

        results
    }

    fn evaluate_filter(&self, filter: &Expr) -> ILResult<Vec<Uuid>> {
        match filter {
            // TODO: add between expr && combine expr && is null expr / is not null expr?
            Expr::BinaryExpr(binary_expr) => {
                let Expr::Column(_) = binary_expr.left.as_ref() else {
                    return Err(ILError::index(
                        "Left side of binary expression must be a column",
                    ));
                };
                let Expr::Literal(literal) = binary_expr.right.as_ref() else {
                    return Err(ILError::index(
                        "Right side of binary expression must be a literal",
                    ));
                };

                let value = &OrderedScalar(literal.value.clone());
                match binary_expr.op {
                    BinaryOp::Eq => Ok(self.point_query(value)),
                    BinaryOp::Lt => Ok(self.range_query((Unbounded, Excluded(value)))),
                    BinaryOp::LtEq => Ok(self.range_query((Unbounded, Included(value)))),
                    BinaryOp::Gt => Ok(self.range_query((Excluded(value), Unbounded))),
                    BinaryOp::GtEq => Ok(self.range_query((Included(value), Unbounded))),
                    _ => Err(ILError::index(
                        "Unsupported binary operation for B-tree index",
                    )),
                }
            }
            _ => Err(ILError::index(
                "Unsupported filter expression for B-tree index",
            )),
        }
    }
}

#[async_trait::async_trait]
impl Index for BTreeIndex {
    async fn search(&self, _query: &dyn SearchQuery) -> ILResult<SearchIndexEntries> {
        Err(ILError::not_supported(
            "B-tree index does not support search",
        ))
    }

    async fn filter(&self, filters: &[Expr]) -> ILResult<FilterIndexEntries> {
        if filters.is_empty() {
            return Ok(FilterIndexEntries {
                row_ids: build_row_id_array(std::iter::empty::<&[u8]>())?,
            });
        }

        let mut result_row_ids = self.evaluate_filter(&filters[0])?;
        for filter in &filters[1..] {
            let filter_row_ids = self.evaluate_filter(filter)?;
            // TODO: optimize this by using a set
            result_row_ids.retain(|id| filter_row_ids.contains(id));
        }

        let row_id_array = build_row_id_array(result_row_ids.iter().map(|id| id.as_bytes()))?;

        Ok(FilterIndexEntries {
            row_ids: row_id_array,
        })
    }
}
