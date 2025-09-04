use std::sync::Arc;

use arrow::array::{ArrayRef, AsArray, BooleanArray, Datum, RecordBatch};
use arrow::datatypes::{DataType, Schema};
use arrow::error::ArrowError;
use derive_visitor::{Drive, DriveMut};

use crate::catalog::{CatalogDatabase, Scalar};
use crate::expr::{ColumnarValue, Expr};
use crate::{ILError, ILResult};

#[derive(Debug, Clone, Copy, Drive, DriveMut, PartialEq, Eq)]
pub enum BinaryOp {
    /// Expressions are equal
    Eq,
    /// Expressions are not equal
    NotEq,
    /// Left side is smaller than right side
    Lt,
    /// Left side is smaller or equal to right side
    LtEq,
    /// Left side is greater than right side
    Gt,
    /// Left side is greater or equal to right side
    GtEq,
    /// Addition
    Plus,
    /// Subtraction
    Minus,
    /// Multiplication operator, like `*`
    Multiply,
    /// Division operator, like `/`
    Divide,
    /// Remainder operator, like `%`
    Modulo,
    /// Logical AND, like `&&`
    And,
    /// Logical OR, like `||`
    Or,
    IsDistinctFrom,
    IsNotDistinctFrom,
}

impl std::fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Eq => write!(f, "="),
            BinaryOp::NotEq => write!(f, "!="),
            BinaryOp::Lt => write!(f, "<"),
            BinaryOp::LtEq => write!(f, "<="),
            BinaryOp::Gt => write!(f, ">"),
            BinaryOp::GtEq => write!(f, ">="),
            BinaryOp::Plus => write!(f, "+"),
            BinaryOp::Minus => write!(f, "-"),
            BinaryOp::Multiply => write!(f, "*"),
            BinaryOp::Divide => write!(f, "/"),
            BinaryOp::Modulo => write!(f, "%"),
            BinaryOp::And => write!(f, "AND"),
            BinaryOp::Or => write!(f, "OR"),
            BinaryOp::IsDistinctFrom => write!(f, "IS DISTINCT FROM"),
            BinaryOp::IsNotDistinctFrom => write!(f, "IS NOT DISTINCT FROM"),
        }
    }
}

/// Binary expression
#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq)]
pub struct BinaryExpr {
    /// Left-hand side of the expression
    pub left: Box<Expr>,
    /// The comparison operator
    pub op: BinaryOp,
    /// Right-hand side of the expression
    pub right: Box<Expr>,
}

impl BinaryExpr {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        let left_sql = self.left.to_sql(database)?;
        let right_sql = self.right.to_sql(database)?;
        Ok(format!("({} {} {})", left_sql, self.op, right_sql))
    }

    pub(crate) fn eval(&self, batch: &RecordBatch) -> ILResult<ColumnarValue> {
        let lhs = self.left.eval(batch)?;
        let rhs = self.right.eval(batch)?;

        match self.op {
            BinaryOp::Eq => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::eq),
            BinaryOp::NotEq => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::neq),
            BinaryOp::Lt => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::lt),
            BinaryOp::LtEq => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::lt_eq),
            BinaryOp::Gt => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::gt),
            BinaryOp::GtEq => apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::gt_eq),
            BinaryOp::Plus => apply(&lhs, &rhs, arrow::compute::kernels::numeric::add_wrapping),
            BinaryOp::Minus => apply(&lhs, &rhs, arrow::compute::kernels::numeric::sub_wrapping),
            BinaryOp::Multiply => apply(&lhs, &rhs, arrow::compute::kernels::numeric::mul_wrapping),
            BinaryOp::Divide => apply(&lhs, &rhs, arrow::compute::kernels::numeric::div),
            BinaryOp::Modulo => apply(&lhs, &rhs, arrow::compute::kernels::numeric::rem),
            BinaryOp::And => apply_boolean(&lhs, &rhs, arrow::compute::kernels::boolean::and),
            BinaryOp::Or => apply_boolean(&lhs, &rhs, arrow::compute::kernels::boolean::or),
            BinaryOp::IsDistinctFrom => {
                apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::distinct)
            }
            BinaryOp::IsNotDistinctFrom => {
                apply_cmp(&lhs, &rhs, arrow::compute::kernels::cmp::not_distinct)
            }
        }
    }

    pub fn data_type(&self, schema: &Schema) -> ILResult<DataType> {
        let left_type = self.left.data_type(schema)?;
        let right_type = self.right.data_type(schema)?;
        match self.op {
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::IsDistinctFrom
            | BinaryOp::IsNotDistinctFrom => Ok(DataType::Boolean),
            BinaryOp::Plus
            | BinaryOp::Minus
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo => {
                if left_type == right_type {
                    Ok(left_type)
                } else {
                    Err(ILError::internal(format!(
                        "Cannot get data type of {} {} {}",
                        left_type, self.op, right_type
                    )))
                }
            }
        }
    }
}

impl std::fmt::Display for BinaryExpr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({} {} {})", self.left, self.op, self.right)
    }
}

pub fn apply(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    f: impl Fn(&dyn Datum, &dyn Datum) -> Result<ArrayRef, ArrowError>,
) -> ILResult<ColumnarValue> {
    match (&lhs, &rhs) {
        (ColumnarValue::Array(left), ColumnarValue::Array(right)) => {
            Ok(ColumnarValue::Array(f(&left.as_ref(), &right.as_ref())?))
        }
        (ColumnarValue::Scalar(left), ColumnarValue::Array(right)) => Ok(ColumnarValue::Array(f(
            &left.to_arrow_scalar()?,
            &right.as_ref(),
        )?)),
        (ColumnarValue::Array(left), ColumnarValue::Scalar(right)) => Ok(ColumnarValue::Array(f(
            &left.as_ref(),
            &right.to_arrow_scalar()?,
        )?)),
        (ColumnarValue::Scalar(left), ColumnarValue::Scalar(right)) => {
            let array = f(&left.to_arrow_scalar()?, &right.to_arrow_scalar()?)?;
            let scalar = Scalar::try_from_array(array.as_ref(), 0)?;
            Ok(ColumnarValue::Scalar(scalar))
        }
    }
}

/// Applies a binary [`Datum`] comparison kernel `f` to `lhs` and `rhs`
pub fn apply_cmp(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    f: impl Fn(&dyn Datum, &dyn Datum) -> Result<BooleanArray, ArrowError>,
) -> ILResult<ColumnarValue> {
    apply(lhs, rhs, |l, r| Ok(Arc::new(f(l, r)?)))
}

pub fn apply_boolean(
    lhs: &ColumnarValue,
    rhs: &ColumnarValue,
    f: impl Fn(&BooleanArray, &BooleanArray) -> Result<BooleanArray, ArrowError>,
) -> ILResult<ColumnarValue> {
    match (&lhs, &rhs) {
        (ColumnarValue::Array(left), ColumnarValue::Array(right)) => {
            let left_bool_arr = left.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!("Expected boolean array, got {}", left.data_type()))
            })?;
            let right_bool_arr = right.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!("Expected boolean array, got {}", right.data_type()))
            })?;
            Ok(ColumnarValue::Array(Arc::new(f(
                left_bool_arr,
                right_bool_arr,
            )?)))
        }
        (ColumnarValue::Scalar(left), ColumnarValue::Array(right)) => {
            let array_size = right.len();
            let left_arr = left.to_array_of_size(array_size)?;
            let left_bool_arr = left_arr.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!(
                    "Expected boolean array, got {}",
                    left_arr.data_type()
                ))
            })?;
            let right_bool_arr = right.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!("Expected boolean array, got {}", right.data_type()))
            })?;
            Ok(ColumnarValue::Array(Arc::new(f(
                left_bool_arr,
                right_bool_arr,
            )?)))
        }
        (ColumnarValue::Array(left), ColumnarValue::Scalar(right)) => {
            let array_size = left.len();
            let right_arr = right.to_array_of_size(array_size)?;
            let left_bool_arr = left.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!("Expected boolean array, got {}", left.data_type()))
            })?;
            let right_bool_arr = right_arr.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!(
                    "Expected boolean array, got {}",
                    right_arr.data_type()
                ))
            })?;
            Ok(ColumnarValue::Array(Arc::new(f(
                left_bool_arr,
                right_bool_arr,
            )?)))
        }
        (ColumnarValue::Scalar(left), ColumnarValue::Scalar(right)) => {
            let array_size = 1;
            let left_arr = left.to_array_of_size(array_size)?;
            let right_arr = right.to_array_of_size(array_size)?;
            let left_bool_arr = left_arr.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!(
                    "Expected boolean array, got {}",
                    left_arr.data_type()
                ))
            })?;
            let right_bool_arr = right_arr.as_boolean_opt().ok_or_else(|| {
                ILError::internal(format!(
                    "Expected boolean array, got {}",
                    right_arr.data_type()
                ))
            })?;
            Ok(ColumnarValue::Array(Arc::new(f(
                left_bool_arr,
                right_bool_arr,
            )?)))
        }
    }
}
