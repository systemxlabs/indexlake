use arrow::{
    array::RecordBatch,
    datatypes::DataType,
};
use derive_visitor::{Drive, DriveMut};

use crate::{
    ILResult,
    catalog::CatalogDatabase,
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
    pub fn new(negated: bool, expr: Box<Expr>, pattern: Box<Expr>, case_insensitive: bool) -> Self {
        Self {
            negated,
            case_insensitive,
            expr,
            pattern,
        }
    }
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        let expr = self.expr.to_sql(database)?;
        let pattern = self.pattern.to_sql(database)?;
        match (self.negated, self.case_insensitive) {
            (true, true) => Ok(format!("{} NOT ILIKE {}", expr, pattern)),
            (true, false) => Ok(format!("{} NOT LIKE {}", expr, pattern)),
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

    #[allow(unused)]
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
