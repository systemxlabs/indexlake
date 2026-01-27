use arrow::datatypes::SchemaRef;
use datafusion::prelude::SessionContext;
use datafusion_common::DataFusionError;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{Column, DFSchema, ScalarValue};
use datafusion_expr::Expr;
use datafusion_expr::{ExprSchemable, Operator};
use datafusion_optimizer::analyzer::type_coercion::TypeCoercionRewriter;
use indexlake::catalog::Scalar as ILScalar;
use indexlake::expr::{BinaryOp as ILOperator, Expr as ILExpr};

pub fn datafusion_expr_to_indexlake_expr(
    expr: &Expr,
    schema: &DFSchema,
) -> Result<ILExpr, DataFusionError> {
    match expr {
        Expr::Alias(alias) => datafusion_expr_to_indexlake_expr(&alias.expr, schema),
        Expr::Column(col) => Ok(ILExpr::Column(col.name.clone())),
        Expr::Literal(lit, _) => Ok(datafusion_scalar_to_indexlake_scalar(lit)?.into()),
        Expr::BinaryExpr(binary) => {
            let left = Box::new(datafusion_expr_to_indexlake_expr(&binary.left, schema)?);
            let op = datafusion_operator_to_indexlake_operator(&binary.op)?;
            let right = Box::new(datafusion_expr_to_indexlake_expr(&binary.right, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left,
                op,
                right,
            }))
        }
        Expr::Not(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::Not(expr))
        }
        Expr::IsNull(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::IsNull(expr))
        }
        Expr::IsNotNull(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::IsNotNull(expr))
        }
        Expr::IsTrue(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsNotDistinctFrom,
                right: Box::new(indexlake::expr::lit(true)),
            }))
        }
        Expr::IsNotTrue(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsDistinctFrom,
                right: Box::new(indexlake::expr::lit(true)),
            }))
        }
        Expr::IsFalse(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsNotDistinctFrom,
                right: Box::new(indexlake::expr::lit(false)),
            }))
        }
        Expr::IsNotFalse(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsDistinctFrom,
                right: Box::new(indexlake::expr::lit(false)),
            }))
        }
        Expr::IsUnknown(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsNotDistinctFrom,
                right: Box::new(ILScalar::Boolean(None).into()),
            }))
        }
        Expr::IsNotUnknown(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: expr,
                op: ILOperator::IsDistinctFrom,
                right: Box::new(ILScalar::Boolean(None).into()),
            }))
        }
        Expr::Between(between) => {
            let expr = datafusion_expr_to_indexlake_expr(&between.expr, schema)?;
            let low = datafusion_expr_to_indexlake_expr(&between.low, schema)?;
            let high = datafusion_expr_to_indexlake_expr(&between.high, schema)?;

            let left_expr = expr.clone().gteq(low);
            let right_expr = expr.lteq(high);
            let and_expr = left_expr.and(right_expr);
            if between.negated {
                Ok(ILExpr::Not(Box::new(and_expr)))
            } else {
                Ok(and_expr)
            }
        }
        Expr::InList(in_list) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(&in_list.expr, schema)?);
            let list = in_list
                .list
                .iter()
                .map(|expr| datafusion_expr_to_indexlake_expr(expr, schema))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ILExpr::InList(indexlake::expr::InList {
                expr,
                list,
                negated: in_list.negated,
            }))
        }
        Expr::Like(like) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(&like.expr, schema)?);
            let pattern = Box::new(datafusion_expr_to_indexlake_expr(&like.pattern, schema)?);
            Ok(ILExpr::Like(indexlake::expr::Like {
                expr,
                pattern,
                negated: like.negated,
                case_insensitive: like.case_insensitive,
            }))
        }
        Expr::Cast(cast) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(&cast.expr, schema)?);
            Ok(ILExpr::Cast(indexlake::expr::Cast {
                expr,
                cast_type: cast.data_type.clone(),
            }))
        }
        Expr::TryCast(try_cast) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(&try_cast.expr, schema)?);
            Ok(ILExpr::TryCast(indexlake::expr::TryCast {
                expr,
                cast_type: try_cast.data_type.clone(),
            }))
        }
        Expr::Negative(expr) => {
            let expr = Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?);
            Ok(ILExpr::Negative(expr))
        }
        Expr::Case(case) => {
            let when_then = match &case.expr {
                Some(expr) => {
                    let expr = datafusion_expr_to_indexlake_expr(expr, schema)?;
                    case.when_then_expr
                        .iter()
                        .map(|(when, then)| {
                            let when = expr
                                .clone()
                                .eq(datafusion_expr_to_indexlake_expr(when, schema)?);
                            let then = datafusion_expr_to_indexlake_expr(then, schema)?;
                            Ok::<_, DataFusionError>((Box::new(when), Box::new(then)))
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                None => case
                    .when_then_expr
                    .iter()
                    .map(|(when, then)| {
                        Ok::<_, DataFusionError>((
                            Box::new(datafusion_expr_to_indexlake_expr(when, schema)?),
                            Box::new(datafusion_expr_to_indexlake_expr(then, schema)?),
                        ))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            };
            let else_expr = match &case.else_expr {
                Some(expr) => Some(Box::new(datafusion_expr_to_indexlake_expr(expr, schema)?)),
                None => None,
            };
            Ok(ILExpr::Case(indexlake::expr::Case {
                when_then,
                else_expr,
            }))
        }
        Expr::ScalarFunction(func) => {
            let mut arg_types = Vec::with_capacity(func.args.len());
            for arg in func.args.iter() {
                let (data_type, _) = arg.data_type_and_nullable(schema)?;
                arg_types.push(data_type);
            }
            let args = func
                .args
                .iter()
                .map(|arg| datafusion_expr_to_indexlake_expr(arg, schema))
                .collect::<Result<Vec<_>, _>>()?;
            let name = func.name().to_string();
            let return_type = func.func.return_type(&arg_types)?;
            Ok(ILExpr::Function(indexlake::expr::Function {
                name,
                args,
                return_type,
            }))
        }
        _ => Err(DataFusionError::NotImplemented(format!(
            "Unsupported expr: {expr}"
        ))),
    }
}

pub fn datafusion_scalar_to_indexlake_scalar(
    scalar: &ScalarValue,
) -> Result<ILScalar, DataFusionError> {
    match scalar {
        ScalarValue::Boolean(v) => Ok(ILScalar::Boolean(*v)),
        ScalarValue::Int8(v) => Ok(ILScalar::Int8(*v)),
        ScalarValue::Int16(v) => Ok(ILScalar::Int16(*v)),
        ScalarValue::Int32(v) => Ok(ILScalar::Int32(*v)),
        ScalarValue::Int64(v) => Ok(ILScalar::Int64(*v)),
        ScalarValue::UInt8(v) => Ok(ILScalar::UInt8(*v)),
        ScalarValue::UInt16(v) => Ok(ILScalar::UInt16(*v)),
        ScalarValue::UInt32(v) => Ok(ILScalar::UInt32(*v)),
        ScalarValue::UInt64(v) => Ok(ILScalar::UInt64(*v)),
        ScalarValue::Float32(v) => Ok(ILScalar::Float32(*v)),
        ScalarValue::Float64(v) => Ok(ILScalar::Float64(*v)),
        ScalarValue::Utf8(v) => Ok(ILScalar::Utf8(v.clone())),
        ScalarValue::Utf8View(v) => Ok(ILScalar::Utf8View(v.clone())),
        ScalarValue::LargeUtf8(v) => Ok(ILScalar::LargeUtf8(v.clone())),
        ScalarValue::Binary(v) => Ok(ILScalar::Binary(v.clone())),
        ScalarValue::BinaryView(v) => Ok(ILScalar::BinaryView(v.clone())),
        ScalarValue::FixedSizeBinary(s, v) => Ok(ILScalar::FixedSizeBinary(*s, v.clone())),
        ScalarValue::LargeBinary(v) => Ok(ILScalar::LargeBinary(v.clone())),
        ScalarValue::TimestampSecond(v, tz) => Ok(ILScalar::TimestampSecond(*v, tz.clone())),
        ScalarValue::TimestampMillisecond(v, tz) => {
            Ok(ILScalar::TimestampMillisecond(*v, tz.clone()))
        }
        ScalarValue::TimestampMicrosecond(v, tz) => {
            Ok(ILScalar::TimestampMicrosecond(*v, tz.clone()))
        }
        ScalarValue::TimestampNanosecond(v, tz) => {
            Ok(ILScalar::TimestampNanosecond(*v, tz.clone()))
        }
        ScalarValue::Date32(v) => Ok(ILScalar::Date32(*v)),
        ScalarValue::Date64(v) => Ok(ILScalar::Date64(*v)),
        ScalarValue::Time32Second(v) => Ok(ILScalar::Time32Second(*v)),
        ScalarValue::Time32Millisecond(v) => Ok(ILScalar::Time32Millisecond(*v)),
        ScalarValue::Time64Microsecond(v) => Ok(ILScalar::Time64Microsecond(*v)),
        ScalarValue::Time64Nanosecond(v) => Ok(ILScalar::Time64Nanosecond(*v)),
        ScalarValue::List(v) => Ok(ILScalar::List(v.clone())),
        ScalarValue::FixedSizeList(v) => Ok(ILScalar::FixedSizeList(v.clone())),
        ScalarValue::LargeList(v) => Ok(ILScalar::LargeList(v.clone())),
        ScalarValue::Decimal128(v, p, s) => Ok(ILScalar::Decimal128(*v, *p, *s)),
        ScalarValue::Decimal256(v, p, s) => Ok(ILScalar::Decimal256(*v, *p, *s)),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Unsupported scalar: {scalar}"
        ))),
    }
}

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
        ILExpr::Cast(cast) => Ok(Expr::Cast(datafusion_expr::expr::Cast {
            expr: Box::new(indexlake_expr_to_datafusion_expr(&cast.expr)?),
            data_type: cast.cast_type.clone(),
        })),
        ILExpr::TryCast(try_cast) => Ok(Expr::TryCast(datafusion_expr::expr::TryCast {
            expr: Box::new(indexlake_expr_to_datafusion_expr(&try_cast.expr)?),
            data_type: try_cast.cast_type.clone(),
        })),
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

pub fn datafusion_operator_to_indexlake_operator(
    operator: &Operator,
) -> Result<ILOperator, DataFusionError> {
    match operator {
        Operator::Eq => Ok(ILOperator::Eq),
        Operator::NotEq => Ok(ILOperator::NotEq),
        Operator::Lt => Ok(ILOperator::Lt),
        Operator::LtEq => Ok(ILOperator::LtEq),
        Operator::Gt => Ok(ILOperator::Gt),
        Operator::GtEq => Ok(ILOperator::GtEq),
        Operator::Plus => Ok(ILOperator::Plus),
        Operator::Minus => Ok(ILOperator::Minus),
        Operator::Multiply => Ok(ILOperator::Multiply),
        Operator::Divide => Ok(ILOperator::Divide),
        Operator::Modulo => Ok(ILOperator::Modulo),
        Operator::And => Ok(ILOperator::And),
        Operator::Or => Ok(ILOperator::Or),
        Operator::IsDistinctFrom => Ok(ILOperator::IsDistinctFrom),
        Operator::IsNotDistinctFrom => Ok(ILOperator::IsNotDistinctFrom),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Unsupported operator: {operator}",
        ))),
    }
}

pub fn parse_expr(expr: &str, schema: SchemaRef) -> Result<ILExpr, DataFusionError> {
    let ctx = SessionContext::new();
    let df_schema = DFSchema::try_from(schema)?;
    let expr = ctx.parse_sql_expr(expr, &df_schema)?;

    let mut rewriter = TypeCoercionRewriter::new(&df_schema);
    let coerced_expr = expr.rewrite(&mut rewriter)?;

    datafusion_expr_to_indexlake_expr(&coerced_expr.data, &df_schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    #[test]
    fn test_parse_expr() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let expr = parse_expr("id + 1", schema).unwrap();
        assert_eq!(
            expr,
            ILExpr::BinaryExpr(indexlake::expr::BinaryExpr {
                left: Box::new(ILExpr::Cast(indexlake::expr::Cast {
                    expr: Box::new(indexlake::expr::col("id")),
                    cast_type: DataType::Int64,
                })),
                right: Box::new(indexlake::expr::lit(1i64)),
                op: ILOperator::Plus,
            })
        );
    }
}
