use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::ops::Bound;
use std::ops::Bound::{Excluded, Included, Unbounded};
use std::sync::Arc;

use arrow::array::{ArrayRef, new_null_array};
use arrow::datatypes::FieldRef;
use indexlake::catalog::Scalar;
use indexlake::expr::{BinaryOp, Expr};
use indexlake::index::{
    FilterIndexEntries, Index, IndexResultColumn, IndexResultOptions, SearchIndexEntries,
    SearchQuery,
};
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
    key_field: FieldRef,
}

impl BTreeIndex {
    pub fn new(key_field: FieldRef) -> Self {
        Self {
            btree: BTreeMap::new(),
            key_field,
        }
    }

    pub fn insert(&mut self, key: OrderedScalar, row_id: Uuid) -> ILResult<()> {
        self.btree.entry(key).or_default().push(row_id);
        Ok(())
    }

    fn point_query(&self, point: &OrderedScalar) -> Vec<(Uuid, Scalar)> {
        self.btree
            .get(point)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|row_id| (row_id, point.0.clone()))
            .collect()
    }

    fn range_query(
        &self,
        range: (Bound<&OrderedScalar>, Bound<&OrderedScalar>),
    ) -> Vec<(Uuid, Scalar)> {
        let mut results = Vec::new();

        for (key, row_ids) in self.btree.range(range) {
            results.extend(
                row_ids
                    .iter()
                    .copied()
                    .map(|row_id| (row_id, key.0.clone())),
            );
        }

        results
    }

    fn evaluate_filter(&self, filter: &Expr) -> ILResult<Vec<(Uuid, Scalar)>> {
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
    async fn search(
        &self,
        _query: &dyn SearchQuery,
        _options: &IndexResultOptions,
    ) -> ILResult<SearchIndexEntries> {
        Err(ILError::not_supported(
            "B-tree index does not support search",
        ))
    }

    async fn filter(
        &self,
        filters: &[Expr],
        options: &IndexResultOptions,
    ) -> ILResult<FilterIndexEntries> {
        if filters.is_empty() {
            return Ok(FilterIndexEntries {
                row_ids: Vec::new(),
                dynamic_columns: Vec::new(),
            });
        }

        let mut result_entries = self.evaluate_filter(&filters[0])?;
        for filter in &filters[1..] {
            let filter_row_ids = self
                .evaluate_filter(filter)?
                .into_iter()
                .map(|(row_id, _)| row_id)
                .collect::<Vec<_>>();
            // TODO: optimize this by using a set
            result_entries.retain(|(row_id, _)| filter_row_ids.contains(row_id));
        }

        let row_ids = result_entries
            .iter()
            .map(|(row_id, _)| *row_id)
            .collect::<Vec<_>>();

        let mut dynamic_columns = Vec::new();
        for column in &options.columns {
            match column.name.as_str() {
                "index_key" => {
                    let values = scalars_to_array(
                        self.key_field.data_type(),
                        result_entries
                            .iter()
                            .map(|(_, value)| value)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )?;
                    dynamic_columns.push(IndexResultColumn {
                        field: Arc::new(
                            self.key_field
                                .as_ref()
                                .clone()
                                .with_name(&column.output_name),
                        ),
                        values,
                    });
                }
                _ => {
                    return Err(ILError::invalid_input(format!(
                        "Unsupported result column `{}` for index kind `btree`",
                        column.name
                    )));
                }
            }
        }

        Ok(FilterIndexEntries {
            row_ids,
            dynamic_columns,
        })
    }
}

fn scalars_to_array(
    data_type: &arrow::datatypes::DataType,
    values: &[&Scalar],
) -> ILResult<ArrayRef> {
    if values.is_empty() {
        return Ok(new_null_array(data_type, 0));
    }
    let arrays = values
        .iter()
        .map(|value| value.to_array_of_size(1))
        .collect::<ILResult<Vec<_>>>()?;
    let arrays = arrays
        .iter()
        .map(|array| array.as_ref())
        .collect::<Vec<_>>();
    Ok(arrow::compute::concat(arrays.as_slice())?)
}
