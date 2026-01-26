mod binary;
mod builder;
mod case;
mod compute;
mod like;
mod text;
mod utils;
mod visitor;

pub use binary::*;
pub use builder::*;
pub use case::*;
pub use compute::*;
pub use like::*;
pub use text::{deserialize_expr, serialize_expr};
pub use utils::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
pub use visitor::*;

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use arrow::array::{ArrayRef, AsArray, BooleanArray, RecordBatch};
use arrow::buffer::BooleanBuffer;
use arrow::compute::CastOptions;
use arrow::datatypes::{DataType, Schema};
use arrow::util::display::{DurationFormat, FormatOptions};

use derive_visitor::{Drive, DriveMut};

use crate::catalog::{INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use crate::{ILError, ILResult};

pub const DEFAULT_CAST_OPTIONS: CastOptions<'static> = CastOptions {
    safe: false,
    format_options: DEFAULT_FORMAT_OPTIONS,
};

pub const DEFAULT_FORMAT_OPTIONS: FormatOptions<'static> =
    FormatOptions::new().with_duration_format(DurationFormat::Pretty);

/// Represents logical expressions such as `A + 1`
#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq, Serialize, Deserialize)]
pub enum Expr {
    /// A named reference
    Column(String),
    /// A constant value
    Literal(Literal),
    /// A binary expression such as "age > 21"
    BinaryExpr(BinaryExpr),
    /// Negation of an expression. The expression's type must be a boolean to
    /// make sense
    Not(Box<Expr>),
    /// True if argument is NULL, false otherwise
    IsNull(Box<Expr>),
    /// True if argument is not NULL, false otherwise
    IsNotNull(Box<Expr>),
    /// Returns whether the list contains the expr value
    InList(InList),
    Function(Function),
    Like(Like),
    Cast(Cast),
    TryCast(TryCast),
    Negative(Box<Expr>),
    Case(Case),
}

impl Expr {
    pub fn eval(&self, batch: &RecordBatch) -> ILResult<ColumnarValue> {
        match self {
            Expr::Column(name) => {
                let index = batch.schema().index_of(name)?;
                Ok(ColumnarValue::Array(batch.column(index).clone()))
            }
            Expr::Literal(literal) => Ok(ColumnarValue::Scalar(literal.value.clone())),
            Expr::BinaryExpr(binary_expr) => binary_expr.eval(batch),
            Expr::Not(expr) => match expr.eval(batch)? {
                ColumnarValue::Array(array) => {
                    let array = array.as_boolean_opt().ok_or_else(|| {
                        ILError::internal(format!(
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
            Expr::Function(_) => Err(ILError::invalid_input(
                "Function can only be used for index",
            )),
            Expr::Like(like) => like.eval(batch),
            Expr::Cast(cast) => {
                let value = cast.expr.eval(batch)?;
                value.cast_to(&cast.cast_type, &DEFAULT_CAST_OPTIONS)
            }
            Expr::TryCast(try_cast) => {
                let value = try_cast.expr.eval(batch)?;
                let options = CastOptions {
                    safe: true,
                    format_options: DEFAULT_FORMAT_OPTIONS,
                };
                value.cast_to(&try_cast.cast_type, &options)
            }
            Expr::Negative(expr) => match expr.eval(batch)? {
                ColumnarValue::Array(array) => {
                    let result = arrow::compute::kernels::numeric::neg_wrapping(array.as_ref())?;
                    Ok(ColumnarValue::Array(result))
                }
                ColumnarValue::Scalar(scalar) => {
                    Ok(ColumnarValue::Scalar(scalar.arithmetic_negate()?))
                }
            },
            Expr::Case(case) => case.eval(batch),
        }
    }

    /// Evaluate an expression against a RecordBatch after first applying a
    /// validity array
    pub fn eval_selection(
        &self,
        batch: &RecordBatch,
        selection: &BooleanArray,
    ) -> ILResult<ColumnarValue> {
        let tmp_batch = arrow::compute::filter_record_batch(batch, selection)?;

        let tmp_result = self.eval(&tmp_batch)?;

        if batch.num_rows() == tmp_batch.num_rows() {
            // All values from the `selection` filter are true.
            Ok(tmp_result)
        } else if let ColumnarValue::Array(a) = tmp_result {
            scatter(selection, a.as_ref()).map(ColumnarValue::Array)
        } else {
            Ok(tmp_result)
        }
    }

    pub fn check_data_type(&self, schema: &Schema, expected: &DataType) -> ILResult<()> {
        let return_type = self.data_type(schema)?;
        if &return_type != expected {
            return Err(ILError::invalid_input(format!(
                "expr {self} should return {expected} instead of {return_type}"
            )));
        }
        Ok(())
    }

    pub fn condition_eval(&self, batch: &RecordBatch) -> ILResult<BooleanArray> {
        self.check_data_type(&batch.schema(), &DataType::Boolean)?;

        let array = self.eval(batch)?.into_array(batch.num_rows())?;
        let bool_array = array.as_boolean_opt().ok_or_else(|| {
            ILError::internal(format!(
                "condition should return BooleanArray, but got {}",
                array.data_type()
            ))
        })?;
        Ok(bool_array.clone())
    }

    pub(crate) fn constant_eval(&self) -> ILResult<Scalar> {
        if !visited_columns(self).is_empty() {
            return Err(ILError::invalid_input(format!(
                "expr {self} is not constant"
            )));
        }

        static EMPTY_BATCH: LazyLock<RecordBatch> =
            LazyLock::new(|| RecordBatch::new_empty(Arc::new(Schema::empty())));

        let value = self.eval(&EMPTY_BATCH)?;
        match value {
            ColumnarValue::Scalar(scalar) => Ok(scalar),
            ColumnarValue::Array(array) => {
                if array.len() == 1 {
                    Scalar::try_from_array(&array, 0)
                } else {
                    Err(ILError::invalid_input(format!(
                        "expr {self} is not constant"
                    )))
                }
            }
        }
    }

    pub fn as_literal(&self) -> ILResult<&Literal> {
        match self {
            Expr::Literal(literal) => Ok(literal),
            _ => Err(ILError::internal("Expr is not a literal")),
        }
    }

    pub fn data_type(&self, schema: &Schema) -> ILResult<DataType> {
        match self {
            Expr::Column(name) => {
                let index = schema.index_of(name)?;
                Ok(schema.field(index).data_type().clone())
            }
            Expr::Literal(literal) => Ok(literal.value.data_type()),
            Expr::BinaryExpr(binary_expr) => binary_expr.data_type(schema),
            Expr::Not(_) => Ok(DataType::Boolean),
            Expr::IsNull(_) => Ok(DataType::Boolean),
            Expr::IsNotNull(_) => Ok(DataType::Boolean),
            Expr::InList(_) => Ok(DataType::Boolean),
            Expr::Function(function) => Ok(function.return_type.clone()),
            Expr::Like(_) => Ok(DataType::Boolean),
            Expr::Cast(cast) => Ok(cast.cast_type.clone()),
            Expr::TryCast(try_cast) => Ok(try_cast.cast_type.clone()),
            Expr::Negative(expr) => expr.data_type(schema),
            Expr::Case(case) => case.data_type(schema),
        }
    }

    pub(crate) fn only_visit_row_id_column(&self) -> bool {
        visited_columns(self) == [INTERNAL_ROW_ID_FIELD_NAME]
    }

    pub(crate) fn rewrite_columns(
        mut self,
        field_name_id_map: &HashMap<String, Uuid>,
    ) -> ILResult<Expr> {
        let mut replacer = ColumnReplacer::new(field_name_id_map);
        self.drive_mut(&mut replacer);
        if let Some(fail_col) = replacer.fail_col {
            Err(ILError::invalid_input(format!(
                "Column {fail_col} not found"
            )))
        } else {
            Ok(self)
        }
    }
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "{name}"),
            Expr::Literal(literal) => write!(f, "{}", literal.value),
            Expr::BinaryExpr(binary_expr) => write!(f, "{binary_expr}"),
            Expr::Not(expr) => write!(f, "NOT {expr}"),
            Expr::IsNull(expr) => write!(f, "{expr} IS NULL"),
            Expr::IsNotNull(expr) => write!(f, "{expr} IS NOT NULL"),
            Expr::InList(in_list) => {
                write!(
                    f,
                    "{} IN ({})",
                    in_list.expr,
                    in_list
                        .list
                        .iter()
                        .map(|expr| expr.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Expr::Function(function) => {
                write!(
                    f,
                    "{}({})",
                    function.name,
                    function
                        .args
                        .iter()
                        .map(|expr| expr.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Expr::Like(like) => write!(f, "{like}"),
            Expr::Cast(cast) => write!(f, "CAST({} AS {})", cast.expr, cast.cast_type),
            Expr::TryCast(try_cast) => {
                write!(f, "TRY_CAST({} AS {})", try_cast.expr, try_cast.cast_type)
            }
            Expr::Negative(expr) => write!(f, "-{expr}"),
            Expr::Case(case) => write!(f, "{case}"),
        }
    }
}

#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq, Serialize, Deserialize)]
pub struct Literal {
    #[drive(skip)]
    pub value: Scalar,
}

/// InList expression
#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq, Serialize, Deserialize)]
pub struct InList {
    /// The expression to compare
    pub expr: Box<Expr>,
    /// The list of values to compare against
    pub list: Vec<Expr>,
    /// Whether the expression is negated
    pub negated: bool,
}

#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq, Serialize, Deserialize)]
pub struct Function {
    pub name: String,
    pub args: Vec<Expr>,
    #[drive(skip)]
    pub return_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq, Drive, DriveMut, Serialize, Deserialize)]
pub struct Cast {
    /// The expression being cast
    pub expr: Box<Expr>,
    /// The `DataType` the expression will yield
    #[drive(skip)]
    pub cast_type: DataType,
}

#[derive(Debug, Clone, PartialEq, Eq, Drive, DriveMut, Serialize, Deserialize)]
pub struct TryCast {
    pub expr: Box<Expr>,
    #[drive(skip)]
    pub cast_type: DataType,
}

#[derive(Clone, Debug)]
pub enum ColumnarValue {
    /// Array of values
    Array(ArrayRef),
    /// A single value
    Scalar(Scalar),
}

impl ColumnarValue {
    pub fn data_type(&self) -> DataType {
        match self {
            ColumnarValue::Array(array_value) => array_value.data_type().clone(),
            ColumnarValue::Scalar(scalar_value) => scalar_value.data_type(),
        }
    }

    pub fn into_array(self, num_rows: usize) -> ILResult<ArrayRef> {
        Ok(match self {
            ColumnarValue::Array(array) => array,
            ColumnarValue::Scalar(scalar) => scalar.to_array_of_size(num_rows)?,
        })
    }

    pub fn cast_to(
        &self,
        cast_type: &DataType,
        cast_options: &CastOptions<'static>,
    ) -> ILResult<ColumnarValue> {
        match self {
            ColumnarValue::Array(array) => Ok(ColumnarValue::Array(
                arrow::compute::kernels::cast::cast_with_options(array, cast_type, cast_options)?,
            )),
            ColumnarValue::Scalar(scalar) => Ok(ColumnarValue::Scalar(
                scalar.cast_to(cast_type, cast_options)?,
            )),
        }
    }
}
