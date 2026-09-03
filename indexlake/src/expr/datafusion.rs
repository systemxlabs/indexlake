use datafusion_common::{Column, DataFusionError, ScalarValue};
use datafusion_expr::{Expr, Operator};

use crate::catalog::Scalar as ILScalar;
use crate::expr::{BinaryOp as ILOperator, Expr as ILExpr};

/// Converts an indexlake scalar to a datafusion scalar. Used to hand scan
/// filters to datafusion's pruning machinery.
pub fn indexlake_scalar_to_datafusion_scalar(
    scalar: &ILScalar,
) -> Result<ScalarValue, DataFusionError> {
    match scalar {
        ILScalar::Boolean(v) => Ok(ScalarValue::Boolean(*v)),
        ILScalar::Int8(v) => Ok(ScalarValue::Int8(*v)),
        ILScalar::Int16(v) => Ok(ScalarValue::Int16(*v)),
        ILScalar::Int32(v) => Ok(ScalarValue::Int32(*v)),
        ILScalar::Int64(v) => Ok(ScalarValue::Int64(*v)),
        ILScalar::UInt8(v) => Ok(ScalarValue::UInt8(*v)),
        ILScalar::UInt16(v) => Ok(ScalarValue::UInt16(*v)),
        ILScalar::UInt32(v) => Ok(ScalarValue::UInt32(*v)),
        ILScalar::UInt64(v) => Ok(ScalarValue::UInt64(*v)),
        ILScalar::Float32(v) => Ok(ScalarValue::Float32(*v)),
        ILScalar::Float64(v) => Ok(ScalarValue::Float64(*v)),
        ILScalar::Utf8(v) => Ok(ScalarValue::Utf8(v.clone())),
        ILScalar::Utf8View(v) => Ok(ScalarValue::Utf8View(v.clone())),
        ILScalar::LargeUtf8(v) => Ok(ScalarValue::LargeUtf8(v.clone())),
        ILScalar::Binary(v) => Ok(ScalarValue::Binary(v.clone())),
        ILScalar::BinaryView(v) => Ok(ScalarValue::BinaryView(v.clone())),
        ILScalar::FixedSizeBinary(s, v) => Ok(ScalarValue::FixedSizeBinary(*s, v.clone())),
        ILScalar::LargeBinary(v) => Ok(ScalarValue::LargeBinary(v.clone())),
        ILScalar::TimestampSecond(v, tz) => Ok(ScalarValue::TimestampSecond(*v, tz.clone())),
        ILScalar::TimestampMillisecond(v, tz) => {
            Ok(ScalarValue::TimestampMillisecond(*v, tz.clone()))
        }
        ILScalar::TimestampMicrosecond(v, tz) => {
            Ok(ScalarValue::TimestampMicrosecond(*v, tz.clone()))
        }
        ILScalar::TimestampNanosecond(v, tz) => {
            Ok(ScalarValue::TimestampNanosecond(*v, tz.clone()))
        }
        ILScalar::Date32(v) => Ok(ScalarValue::Date32(*v)),
        ILScalar::Date64(v) => Ok(ScalarValue::Date64(*v)),
        ILScalar::Time32Second(v) => Ok(ScalarValue::Time32Second(*v)),
        ILScalar::Time32Millisecond(v) => Ok(ScalarValue::Time32Millisecond(*v)),
        ILScalar::Time64Microsecond(v) => Ok(ScalarValue::Time64Microsecond(*v)),
        ILScalar::Time64Nanosecond(v) => Ok(ScalarValue::Time64Nanosecond(*v)),
        ILScalar::List(v) => Ok(ScalarValue::List(v.clone())),
        ILScalar::FixedSizeList(v) => Ok(ScalarValue::FixedSizeList(v.clone())),
        ILScalar::LargeList(v) => Ok(ScalarValue::LargeList(v.clone())),
        ILScalar::Decimal128(v, p, s) => Ok(ScalarValue::Decimal128(*v, *p, *s)),
        ILScalar::Decimal256(v, p, s) => Ok(ScalarValue::Decimal256(*v, *p, *s)),
    }
}

/// Converts an indexlake expression to a datafusion expression. Returns
/// `NotImplemented` for expression kinds datafusion pruning cannot consume
/// (currently `ILExpr::Function`); callers should treat conversion failure as
/// "cannot prune" and keep the filter for row-level evaluation.
pub fn indexlake_expr_to_datafusion_expr(expr: &ILExpr) -> Result<Expr, DataFusionError> {
    match expr {
        ILExpr::Column(name) => Ok(Expr::Column(Column::from_name(name))),
        ILExpr::Literal(literal) => {
            let scalar_value = indexlake_scalar_to_datafusion_scalar(&literal.value)?;
            Ok(Expr::Literal(scalar_value, None))
        }
        ILExpr::BinaryExpr(binary) => {
            let left = Box::new(indexlake_expr_to_datafusion_expr(&binary.left)?);
            let right = Box::new(indexlake_expr_to_datafusion_expr(&binary.right)?);
            let op = indexlake_operator_to_datafusion_operator(&binary.op)?;
            Ok(Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
                left,
                op,
                right,
            }))
        }
        ILExpr::Not(expr) => Ok(Expr::Not(Box::new(indexlake_expr_to_datafusion_expr(
            expr,
        )?))),
        ILExpr::IsNull(expr) => Ok(Expr::IsNull(Box::new(indexlake_expr_to_datafusion_expr(
            expr,
        )?))),
        ILExpr::IsNotNull(expr) => Ok(Expr::IsNotNull(Box::new(
            indexlake_expr_to_datafusion_expr(expr)?,
        ))),
        ILExpr::InList(in_list) => {
            let expr = Box::new(indexlake_expr_to_datafusion_expr(&in_list.expr)?);
            let list = in_list
                .list
                .iter()
                .map(indexlake_expr_to_datafusion_expr)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Expr::InList(datafusion_expr::expr::InList {
                expr,
                list,
                negated: in_list.negated,
            }))
        }
        ILExpr::Like(like) => Ok(Expr::Like(datafusion_expr::expr::Like {
            negated: like.negated,
            expr: Box::new(indexlake_expr_to_datafusion_expr(&like.expr)?),
            pattern: Box::new(indexlake_expr_to_datafusion_expr(&like.pattern)?),
            escape_char: None,
            case_insensitive: like.case_insensitive,
        })),
        ILExpr::Cast(cast) => Ok(Expr::Cast(datafusion_expr::expr::Cast::new(
            Box::new(indexlake_expr_to_datafusion_expr(&cast.expr)?),
            cast.cast_type.clone(),
        ))),
        ILExpr::TryCast(try_cast) => Ok(Expr::TryCast(datafusion_expr::expr::TryCast::new(
            Box::new(indexlake_expr_to_datafusion_expr(&try_cast.expr)?),
            try_cast.cast_type.clone(),
        ))),
        ILExpr::Negative(expr) => Ok(Expr::Negative(Box::new(indexlake_expr_to_datafusion_expr(
            expr,
        )?))),
        ILExpr::Case(case) => {
            let when_then_expr = case
                .when_then
                .iter()
                .map(|(when, then)| {
                    Ok::<_, DataFusionError>((
                        Box::new(indexlake_expr_to_datafusion_expr(when)?),
                        Box::new(indexlake_expr_to_datafusion_expr(then)?),
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let else_expr = match &case.else_expr {
                Some(expr) => Some(Box::new(indexlake_expr_to_datafusion_expr(expr)?)),
                None => None,
            };
            Ok(Expr::Case(datafusion_expr::expr::Case {
                expr: None,
                when_then_expr,
                else_expr,
            }))
        }
        ILExpr::Function(_) => Err(DataFusionError::NotImplemented(
            "Unsupported expr: function".to_string(),
        )),
    }
}

pub fn indexlake_operator_to_datafusion_operator(
    operator: &ILOperator,
) -> Result<Operator, DataFusionError> {
    match operator {
        ILOperator::Eq => Ok(Operator::Eq),
        ILOperator::NotEq => Ok(Operator::NotEq),
        ILOperator::Lt => Ok(Operator::Lt),
        ILOperator::LtEq => Ok(Operator::LtEq),
        ILOperator::Gt => Ok(Operator::Gt),
        ILOperator::GtEq => Ok(Operator::GtEq),
        ILOperator::Plus => Ok(Operator::Plus),
        ILOperator::Minus => Ok(Operator::Minus),
        ILOperator::Multiply => Ok(Operator::Multiply),
        ILOperator::Divide => Ok(Operator::Divide),
        ILOperator::Modulo => Ok(Operator::Modulo),
        ILOperator::And => Ok(Operator::And),
        ILOperator::Or => Ok(Operator::Or),
        ILOperator::IsDistinctFrom => Ok(Operator::IsDistinctFrom),
        ILOperator::IsNotDistinctFrom => Ok(Operator::IsNotDistinctFrom),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{col, lit};

    #[test]
    fn converts_binary_predicate() {
        let expr = indexlake_expr_to_datafusion_expr(&col("age").gt(lit(21i64))).unwrap();
        assert_eq!(
            expr,
            Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr {
                left: Box::new(Expr::Column(Column::from_name("age"))),
                op: Operator::Gt,
                right: Box::new(Expr::Literal(ScalarValue::Int64(Some(21)), None)),
            })
        );
    }

    /// Row-id literals are FixedSizeBinary(16); the scalar mapping must keep
    /// them intact so index-scan row-id filters survive conversion.
    #[test]
    fn converts_fixed_size_binary_scalar() {
        let value = vec![7u8; 16];
        let scalar = indexlake_scalar_to_datafusion_scalar(&ILScalar::FixedSizeBinary(
            16,
            Some(value.clone()),
        ))
        .unwrap();
        assert_eq!(scalar, ScalarValue::FixedSizeBinary(16, Some(value)));
    }

    /// Function expressions have no datafusion counterpart; conversion must
    /// fail (so pruning is abandoned) rather than silently drop the filter.
    #[test]
    fn rejects_function_expressions() {
        let expr = indexlake_expr_to_datafusion_expr(&ILExpr::Function(crate::expr::Function {
            name: "version".to_string(),
            args: vec![],
            return_type: arrow::datatypes::DataType::Int64,
        }));
        assert!(expr.is_err());
    }
}
