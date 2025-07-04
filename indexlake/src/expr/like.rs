use arrow::{array::RecordBatch, datatypes::DataType};
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
        let expr = self.expr.to_sql(database.clone())?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::CatalogDatabase;
    use crate::expr::{col, lit};
    use arrow::array::{BooleanArray, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    fn create_batch() -> RecordBatch {
        let schema = Schema::new(vec![Field::new("c1", DataType::Utf8, false)]);
        let c1 = StringArray::from(vec!["hello", "world", "HELLO", "WORLD"]);
        RecordBatch::try_new(Arc::new(schema), vec![Arc::new(c1)]).unwrap()
    }

    #[tokio::test]
    async fn test_like_to_sql() {
        let db = CatalogDatabase::Sqlite;
        let expr = col("c1");
        let pattern = lit("a%".to_string());

        // LIKE
        let like_expr = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            false,
        );
        assert_eq!(like_expr.to_sql(db).unwrap(), "`c1` LIKE 'a%'");

        // NOT LIKE
        let not_like_expr = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            false,
        );
        assert_eq!(not_like_expr.to_sql(db).unwrap(), "`c1` NOT LIKE 'a%'");

        // ILIKE
        let ilike_expr = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            true,
        );
        assert_eq!(ilike_expr.to_sql(db).unwrap(), "`c1` ILIKE 'a%'");

        // NOT ILIKE
        let not_ilike_expr = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            true,
        );
        assert_eq!(not_ilike_expr.to_sql(db).unwrap(), "`c1` NOT ILIKE 'a%'");
    }

    #[test]
    fn test_like_eval() {
        let batch = create_batch();
        let num_rows = batch.num_rows();
        let expr = col("c1");

        // === LIKE (case-sensitive) ===
        // Test LIKE: c1 LIKE 'h%' -> ['hello'] -> [true, false, false, false]
        let pattern_h = lit("h%".to_string());
        let like_expr_h = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_h.clone()),
            false,
        );
        let result = like_expr_h
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![true, false, false, false])
        );

        // Test NOT LIKE: c1 NOT LIKE 'h%' -> ['world', 'HELLO', 'WORLD'] -> [false, true, true, true]
        let not_like_expr_h = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern_h.clone()),
            false,
        );
        let result = not_like_expr_h
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, true, true])
        );

        // === ILIKE (case-insensitive) ===
        // Test ILIKE: c1 ILIKE 'h%' -> ['hello', 'HELLO'] -> [true, false, true, false]
        let ilike_expr_h = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_h.clone()),
            true,
        );
        let result = ilike_expr_h
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![true, false, true, false])
        );

        // Test NOT ILIKE: c1 NOT ILIKE 'h%' -> ['world', 'WORLD'] -> [false, true, false, true]
        let not_ilike_expr_h = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern_h.clone()),
            true,
        );
        let result = not_ilike_expr_h
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, false, true])
        );

        // === More wildcards ===
        // Test with wildcard '%' at the start: c1 LIKE '%d' -> ['world'] -> [false, true, false, false]
        let pattern_d = lit("%d".to_string());
        let like_expr_d = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_d.clone()),
            false,
        );
        let result = like_expr_d
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, false, false])
        );

        // Test with wildcard '%' at the start (case-insensitive): c1 ILIKE '%d' -> ['world', 'WORLD'] -> [false, true, false, true]
        let ilike_expr_d = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_d.clone()),
            true,
        );
        let result = ilike_expr_d
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, false, true])
        );

        // Test with wildcard '_': c1 LIKE 'w_rld' -> ['world'] -> [false, true, false, false]
        let pattern_w = lit("w_rld".to_string());
        let like_expr_w = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_w.clone()),
            false,
        );
        let result = like_expr_w
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, false, false])
        );

        // Test with wildcard '_' (case-insensitive): c1 ILIKE 'W_RLD' -> ['world', 'WORLD'] -> [false, true, false, true]
        let pattern_w = lit("W_RLD".to_string());
        let ilike_expr_w = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern_w.clone()),
            true,
        );
        let result = ilike_expr_w
            .eval(&batch)
            .unwrap()
            .into_array(num_rows)
            .unwrap();
        let result_bool = result.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            result_bool,
            &BooleanArray::from(vec![false, true, false, true])
        );
    }

    #[test]
    fn test_like_display() {
        let expr = col("c1");
        let pattern = lit("a%".to_string());

        // LIKE
        let like_expr = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            false,
        );
        assert_eq!(format!("{}", like_expr), "c1 LIKE a%");

        // NOT LIKE
        let not_like_expr = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            false,
        );
        assert_eq!(format!("{}", not_like_expr), "c1 NOT LIKE a%");

        // ILIKE
        let ilike_expr = LikeExpr::new(
            false,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            true,
        );
        assert_eq!(format!("{}", ilike_expr), "c1 ILIKE a%");

        // NOT ILIKE
        let not_ilike_expr = LikeExpr::new(
            true,
            Box::new(expr.clone()),
            Box::new(pattern.clone()),
            true,
        );
        assert_eq!(format!("{}", not_ilike_expr), "c1 NOT ILIKE a%");
    }

    #[test]
    fn test_like_data_type() {
        let expr = col("c1");
        let pattern = lit("a%".to_string());
        let like_expr = LikeExpr::new(false, Box::new(expr), Box::new(pattern), false);
        assert_eq!(like_expr.data_type().unwrap(), DataType::Boolean);
    }
}
