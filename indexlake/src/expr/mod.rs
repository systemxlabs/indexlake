mod binary;
mod builder;
mod compute;
mod like;
mod utils;
mod visitor;

pub use binary::*;
pub use builder::*;
pub use compute::*;
pub use utils::*;
pub use visitor::*;

use std::sync::Arc;

use arrow::{
    array::{ArrayRef, AsArray, BooleanArray, RecordBatch},
    buffer::BooleanBuffer,
    datatypes::{DataType, Schema},
    error::ArrowError,
};
use parquet::arrow::{ArrowSchemaConverter, ProjectionMask, arrow_reader::ArrowPredicate};

use derive_visitor::{Drive, DriveMut};

use crate::{
    ILError, ILResult,
    catalog::{CatalogDatabase, Row, Scalar},
    expr::like::LikeExpr,
};

/// Represents logical expressions such as `A + 1`
#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq)]
pub enum Expr {
    /// A named reference
    Column(String),
    /// A constant value
    Literal(Scalar),
    /// A binary expression such as "age > 21"
    BinaryExpr(BinaryExpr),
    /// Negation of an expression. The expression's type must be a boolean to make sense
    Not(Box<Expr>),
    /// True if argument is NULL, false otherwise
    IsNull(Box<Expr>),
    /// True if argument is not NULL, false otherwise
    IsNotNull(Box<Expr>),
    /// Returns whether the list contains the expr value
    InList(InList),
    Function(Function),
    LikeExpr(LikeExpr),
}

impl Expr {
    pub fn eval(&self, batch: &RecordBatch) -> ILResult<ColumnarValue> {
        match self {
            Expr::Column(name) => {
                let index = batch.schema().index_of(name)?;
                Ok(ColumnarValue::Array(batch.column(index).clone()))
            }
            Expr::Literal(scalar) => Ok(ColumnarValue::Scalar(scalar.clone())),
            Expr::BinaryExpr(binary_expr) => binary_expr.eval(batch),
            Expr::Not(expr) => match expr.eval(batch)? {
                ColumnarValue::Array(array) => {
                    let array = array.as_boolean_opt().ok_or_else(|| {
                        ILError::InternalError(format!(
                            "Expected boolean array, got {}",
                            array.data_type()
                        ))
                    })?;
                    Ok(ColumnarValue::Array(Arc::new(
                        arrow::compute::kernels::boolean::not(array)?,
                    )))
                }
                ColumnarValue::Scalar(scalar) => {
                    let bool_value = scalar.as_bool()?;
                    let not_bool_value = bool_value.map(|b| !b);
                    Ok(ColumnarValue::Scalar(Scalar::Boolean(not_bool_value)))
                }
            },
            Expr::IsNull(expr) => match expr.eval(batch)? {
                ColumnarValue::Array(array) => Ok(ColumnarValue::Array(Arc::new(
                    arrow::compute::is_null(&array)?,
                ))),
                ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(Scalar::Boolean(Some(
                    scalar.is_null(),
                )))),
            },
            Expr::IsNotNull(expr) => match expr.eval(batch)? {
                ColumnarValue::Array(array) => Ok(ColumnarValue::Array(Arc::new(
                    arrow::compute::is_not_null(&array)?,
                ))),
                ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(Scalar::Boolean(Some(
                    !scalar.is_null(),
                )))),
            },
            Expr::InList(in_list) => {
                let num_rows = batch.num_rows();
                let value = in_list.expr.eval(batch)?.into_array(num_rows)?;
                let is_nested = value.data_type().is_nested();
                let found = in_list.list.iter().map(|expr| expr.eval(batch)).try_fold(
                    BooleanArray::new(BooleanBuffer::new_unset(num_rows), None),
                    |result, expr| -> ILResult<BooleanArray> {
                        let rhs = compare_with_eq(&value, &expr?.into_array(num_rows)?, is_nested)?;
                        Ok(arrow::compute::or_kleene(&result, &rhs)?)
                    },
                )?;

                let r = if in_list.negated {
                    arrow::compute::not(&found)?
                } else {
                    found
                };

                Ok(ColumnarValue::Array(Arc::new(r)))
            }
            Expr::Function(_) => Err(ILError::InvalidInput(
                "Function can only be used for index".to_string(),
            )),
            Expr::LikeExpr(like_expr) => like_expr.eval(batch),
        }
    }

    pub fn data_type(&self, schema: &Schema) -> ILResult<DataType> {
        match self {
            Expr::Column(name) => {
                let index = schema.index_of(name)?;
                Ok(schema.field(index).data_type().clone())
            }
            Expr::Literal(scalar) => Ok(scalar.data_type()),
            Expr::BinaryExpr(binary_expr) => binary_expr.data_type(schema),
            Expr::Not(_) => Ok(DataType::Boolean),
            Expr::IsNull(_) => Ok(DataType::Boolean),
            Expr::IsNotNull(_) => Ok(DataType::Boolean),
            Expr::InList(_) => Ok(DataType::Boolean),
            Expr::Function(function) => Ok(function.return_type.clone()),
            Expr::LikeExpr(like_expr) => like_expr.data_type(),
        }
    }

    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        match self {
            Expr::Column(name) => Ok(database.sql_identifier(name)),
            Expr::Literal(scalar) => Ok(scalar.to_sql(database)),
            Expr::BinaryExpr(binary_expr) => binary_expr.to_sql(database),
            Expr::Not(expr) => Ok(format!("NOT {}", expr.to_sql(database)?)),
            Expr::IsNull(expr) => Ok(format!("{} IS NULL", expr.to_sql(database)?)),
            Expr::IsNotNull(expr) => Ok(format!("{} IS NOT NULL", expr.to_sql(database)?)),
            Expr::InList(in_list) => {
                let list = in_list
                    .list
                    .iter()
                    .map(|expr| expr.to_sql(database))
                    .collect::<ILResult<Vec<_>>>()?
                    .join(", ");
                Ok(format!("{} IN ({})", in_list.expr.to_sql(database)?, list))
            }
            Expr::Function(_) => Err(ILError::InvalidInput(
                "Function can only be used for index".to_string(),
            )),
            Expr::LikeExpr(like_expr) => like_expr.to_sql(database),
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "{}", name),
            Expr::Literal(scalar) => write!(f, "{}", scalar),
            Expr::BinaryExpr(binary_expr) => write!(f, "{}", binary_expr),
            Expr::Not(expr) => write!(f, "NOT {}", expr),
            Expr::IsNull(expr) => write!(f, "{} IS NULL", expr),
            Expr::IsNotNull(expr) => write!(f, "{} IS NOT NULL", expr),
            Expr::InList(in_list) => write!(
                f,
                "{} IN ({})",
                in_list.expr,
                in_list
                    .list
                    .iter()
                    .map(|expr| expr.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Expr::Function(function) => write!(
                f,
                "{}({})",
                function.name,
                function
                    .args
                    .iter()
                    .map(|expr| expr.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Expr::LikeExpr(like_expr) => write!(f, "{}", like_expr),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ExprPredicate {
    expr: Expr,
    projection: ProjectionMask,
}

impl ExprPredicate {
    pub(crate) fn try_new(expr: Expr, projection: ProjectionMask) -> ILResult<Self> {
        Ok(Self { expr, projection })
    }
}

impl ArrowPredicate for ExprPredicate {
    fn projection(&self) -> &ProjectionMask {
        &self.projection
    }

    fn evaluate(&mut self, batch: RecordBatch) -> Result<BooleanArray, ArrowError> {
        let expr_value = self
            .expr
            .eval(&batch)
            .map_err(|e| ArrowError::from_external_error(Box::new(e)))?;
        let array = expr_value
            .into_array(batch.num_rows())
            .map_err(|e| ArrowError::from_external_error(Box::new(e)))?;
        let bool_array = array.as_boolean_opt().ok_or_else(|| {
            ArrowError::ComputeError(format!(
                "ExprPredicate evaluation expected boolean array, got {}",
                array.data_type()
            ))
        })?;

        Ok(bool_array.clone())
    }
}

/// InList expression
#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq)]
pub struct InList {
    /// The expression to compare
    pub expr: Box<Expr>,
    /// The list of values to compare against
    pub list: Vec<Expr>,
    /// Whether the expression is negated
    pub negated: bool,
}

#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq)]
pub struct Function {
    pub name: String,
    pub args: Vec<Expr>,
    #[drive(skip)]
    pub return_type: DataType,
}

#[derive(Clone, Debug)]
pub enum ColumnarValue {
    /// Array of values
    Array(ArrayRef),
    /// A single value
    Scalar(Scalar),
}

impl ColumnarValue {
    pub fn into_array(self, num_rows: usize) -> ILResult<ArrayRef> {
        Ok(match self {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows)?,
        })
    }
}
