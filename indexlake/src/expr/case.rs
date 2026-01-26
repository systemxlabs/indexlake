use std::borrow::Cow;

use arrow::array::{Array, BooleanArray, RecordBatch, new_null_array};
use arrow::compute::kernels::zip::zip;
use arrow::compute::{and, and_not, prep_null_mask_filter};
use arrow_schema::{DataType, Schema};
use derive_visitor::{Drive, DriveMut};
use serde::{Deserialize, Serialize};

use crate::expr::{ColumnarValue, Expr, try_cast};
use crate::{ILError, ILResult};

#[derive(Debug, Clone, Drive, DriveMut, PartialEq, Eq, Serialize, Deserialize)]
pub struct Case {
    pub when_then: Vec<(Box<Expr>, Box<Expr>)>,
    pub else_expr: Option<Box<Expr>>,
}

impl Case {
    pub fn data_type(&self, schema: &Schema) -> ILResult<DataType> {
        // since all then results have the same data type, we can choose any one as the
        // return data type except for the null.
        let mut data_type = DataType::Null;
        for i in 0..self.when_then.len() {
            data_type = self.when_then[i].1.data_type(schema)?;
            if !data_type.equals_datatype(&DataType::Null) {
                break;
            }
        }
        // if all then results are null, we use data type of else expr instead if
        // possible.
        if data_type.equals_datatype(&DataType::Null)
            && let Some(e) = &self.else_expr
        {
            data_type = e.data_type(schema)?;
        }

        Ok(data_type)
    }

    pub fn eval(&self, batch: &RecordBatch) -> ILResult<ColumnarValue> {
        let return_type = self.data_type(&batch.schema())?;

        // start with nulls as default output
        let mut current_value = new_null_array(&return_type, batch.num_rows());
        let mut remainder = BooleanArray::from(vec![true; batch.num_rows()]);
        for i in 0..self.when_then.len() {
            let when_value = self.when_then[i].0.eval_selection(batch, &remainder)?;
            let when_value = when_value.into_array(batch.num_rows())?;
            let when_value = when_value
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    ILError::internal("WHEN expression did not return a BooleanArray")
                })?;
            // Treat 'NULL' as false value
            let when_value = match when_value.null_count() {
                0 => Cow::Borrowed(when_value),
                _ => Cow::Owned(prep_null_mask_filter(when_value)),
            };
            // Make sure we only consider rows that have not been matched yet
            let when_value = and(&when_value, &remainder)?;

            // When no rows available for when clause, skip then clause
            if when_value.true_count() == 0 {
                continue;
            }

            let then_value = self.when_then[i].1.eval_selection(batch, &when_value)?;

            current_value = match then_value {
                ColumnarValue::Scalar(then_value) => {
                    zip(&when_value, &then_value.to_arrow_scalar()?, &current_value)?
                }
                ColumnarValue::Array(then_value) => zip(&when_value, &then_value, &current_value)?,
            };

            // Succeed tuples should be filtered out for short-circuit evaluation,
            // null values for the current when expr should be kept
            remainder = and_not(&remainder, &when_value)?;
        }

        if let Some(e) = &self.else_expr {
            // keep `else_expr`'s data type and return type consistent
            let expr = try_cast(e.as_ref().clone(), &batch.schema(), return_type.clone())?;
            let else_ = expr
                .eval_selection(batch, &remainder)?
                .into_array(batch.num_rows())?;
            current_value = zip(&remainder, &else_, &current_value)?;
        }

        Ok(ColumnarValue::Array(current_value))
    }
}

impl std::fmt::Display for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CASE")?;
        for (when, then) in &self.when_then {
            write!(f, " WHEN {when} THEN {then}")?;
        }
        if let Some(else_expr) = &self.else_expr {
            write!(f, " ELSE {else_expr}")?;
        }
        write!(f, " END")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int32Array, StringArray};
    use arrow_schema::Field;

    use crate::expr::{col, lit};

    use super::*;

    #[test]
    fn test_case_expr() -> Result<(), Box<dyn std::error::Error>> {
        // CASE WHEN a = 'foo' THEN 123 WHEN a = 'bar' THEN 456 ELSE 999 END
        let when_then = vec![
            (Box::new(col("a").eq(lit("foo"))), Box::new(lit(123i32))),
            (Box::new(col("a").eq(lit("bar"))), Box::new(lit(456i32))),
        ];
        let case_expr = Case {
            when_then,
            else_expr: Some(Box::new(lit(999i32))),
        };

        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]);
        let a = StringArray::from(vec![Some("foo"), Some("baz"), None, Some("bar")]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;

        let result = case_expr.eval(&batch)?.into_array(batch.num_rows())?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(result, &Int32Array::from(vec![123, 999, 999, 456]));

        Ok(())
    }

    #[test]
    fn test_case_expr_without_else() -> Result<(), Box<dyn std::error::Error>> {
        // CASE WHEN a = 'foo' THEN 123 WHEN a = 'bar' THEN 456 END
        let when_then = vec![
            (Box::new(col("a").eq(lit("foo"))), Box::new(lit(123i32))),
            (Box::new(col("a").eq(lit("bar"))), Box::new(lit(456i32))),
        ];
        let case_expr = Case {
            when_then,
            else_expr: None,
        };

        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]);
        let a = StringArray::from(vec![Some("foo"), Some("baz"), None, Some("bar")]);
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a)])?;

        let result = case_expr.eval(&batch)?.into_array(batch.num_rows())?;
        let result = result.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(
            result,
            &Int32Array::from(vec![Some(123), None, None, Some(456)])
        );

        Ok(())
    }
}
