use std::sync::Arc;

use arrow::{
    array::{ArrayRef, AsArray, BooleanArray, Datum, RecordBatch},
    datatypes::{DataType, Schema},
    error::ArrowError,
};
use derive_visitor::{Drive, DriveMut};

use crate::{
    ILError, ILResult,
    catalog::{CatalogDatabase, Row, Scalar},
    expr::{ColumnarValue, Expr, apply_cmp},
};

#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq)]
pub struct LikeExpr {
    negated: bool,
    case_insensitive: bool,
    expr: Box<Expr>,
    pattern: Box<Expr>,
}

impl LikeExpr {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        let expr = self.expr.to_sql(database)?;
        let pattern = self.pattern.to_sql(database)?;
        match (self.negated, self.case_insensitive) {
            (true, true) => Ok(format!("NOT ILIKE {}", pattern)),
            (true, false) => Ok(format!("NOT LIKE {}", pattern)),
            (false, true) => Ok(format!("{} ILIKE {}", expr, pattern)),
            (false, false) => Ok(format!("{} LIKE {}", expr, pattern)),
        }
    }

    pub(crate) fn eval(&self, batch: &RecordBatch) -> ILResult<ColumnarValue> {
        use arrow::compute::*;
        let lhs = self.expr.eval(batch)?;
        let rhs = self.pattern.eval(batch)?;
        match (self.negated, self.case_insensitive) {
            (false, false) => apply_cmp(&lhs, &rhs, like),
            (false, true) => apply_cmp(&lhs, &rhs, ilike),
            (true, false) => apply_cmp(&lhs, &rhs, nlike),
            (true, true) => apply_cmp(&lhs, &rhs, nilike),
        }
    }

    pub fn data_type(&self) -> ILResult<DataType> {
        Ok(DataType::Boolean)
    }

    /// Operator name
    fn op_name(&self) -> &str {
        match (self.negated, self.case_insensitive) {
            (false, false) => "LIKE",
            (true, false) => "NOT LIKE",
            (false, true) => "ILIKE",
            (true, true) => "NOT ILIKE",
        }
    }
}

impl std::fmt::Display for LikeExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} {} {}", self.expr, self.op_name(), self.pattern)
    }
}
