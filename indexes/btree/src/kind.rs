use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::DataType;
use indexlake::expr::{BinaryOp, Expr};
use indexlake::index::{
    FilterSupport, IndexBuilder, IndexDefinition, IndexDefinitionRef, IndexKind, IndexParams,
    SearchQuery,
};
use indexlake::{ILError, ILResult};
use serde::{Deserialize, Serialize};

use crate::BTreeIndexBuilder;

#[derive(Debug)]
pub struct BTreeIndexKind;

impl IndexKind for BTreeIndexKind {
    fn kind(&self) -> &str {
        "btree"
    }

    fn decode_params(&self, params: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params: BTreeIndexParams = serde_json::from_str(params)
            .map_err(|e| ILError::index(format!("Failed to decode B-tree index params: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::index(
                "B-tree index requires exactly one key column",
            ));
        }
        let key_column_name = &index_def.key_columns[0];
        let key_field = index_def.table_schema.field_with_name(key_column_name)?;

        match key_field.data_type() {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Timestamp(..)
            | DataType::Date32
            | DataType::Date64
            | DataType::Time32(_)
            | DataType::Time64(_) => Ok(()),
            _ => Err(ILError::index(
                "B-tree index key column must be a comparable data type (integers, strings, timestamps, dates)",
            )),
        }
    }

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(BTreeIndexBuilder::try_new(index_def.clone())?))
    }

    fn supports_search(
        &self,
        _index_def: &IndexDefinition,
        _query: &dyn SearchQuery,
    ) -> ILResult<bool> {
        Ok(false)
    }

    fn supports_filter(
        &self,
        index_def: &IndexDefinition,
        filter: &Expr,
    ) -> ILResult<FilterSupport> {
        match filter {
            Expr::BinaryExpr(binary_expr) => match binary_expr.op {
                BinaryOp::Eq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                    if let Expr::Column(column_name) = binary_expr.left.as_ref()
                        && column_name == &index_def.key_columns[0]
                        && let Expr::Literal(_) = binary_expr.right.as_ref()
                    {
                        Ok(FilterSupport::Exact)
                    } else {
                        Ok(FilterSupport::Unsupported)
                    }
                }
                _ => Ok(FilterSupport::Unsupported),
            },
            _ => Ok(FilterSupport::Unsupported),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BTreeIndexParams {}

impl IndexParams for BTreeIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self) -> ILResult<String> {
        serde_json::to_string(self).map_err(|e| ILError::index(e.to_string()))
    }
}
