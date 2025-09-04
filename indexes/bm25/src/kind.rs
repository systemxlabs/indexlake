use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::DataType;
use indexlake::expr::Expr;
use indexlake::index::{
    FilterSupport, IndexBuilder, IndexDefinition, IndexDefinitionRef, IndexKind, IndexParams,
    SearchQuery,
};
use indexlake::{ILError, ILResult};
use serde::{Deserialize, Serialize};

use crate::{BM25SearchQuery, Bm25IndexBuilder};

#[derive(Debug)]
pub struct BM25IndexKind;

impl IndexKind for BM25IndexKind {
    fn kind(&self) -> &str {
        "bm25"
    }

    fn decode_params(&self, params: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params: BM25IndexParams = serde_json::from_str(params)
            .map_err(|e| ILError::index(format!("Failed to decode BM25 index params: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::index("BM25 index requires exactly one key column"));
        }
        let key_column_name = &index_def.key_columns[0];
        let key_field = index_def.table_schema.field_with_name(key_column_name)?;
        if !matches!(
            key_field.data_type(),
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            return Err(ILError::index(
                "BM25 index key column must be a utf8 / large utf8 / utf8 view column",
            ));
        }
        Ok(())
    }

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(Bm25IndexBuilder::try_new(index_def.clone())?))
    }

    fn supports_search(&self, _: &IndexDefinition, query: &dyn SearchQuery) -> ILResult<bool> {
        if query.as_any().downcast_ref::<BM25SearchQuery>().is_some() {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn supports_filter(&self, _: &IndexDefinition, _: &Expr) -> ILResult<FilterSupport> {
        Ok(FilterSupport::Unsupported)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BM25IndexParams {
    pub avgdl: f32,
}

impl IndexParams for BM25IndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self) -> ILResult<String> {
        serde_json::to_string(self).map_err(|e| ILError::index(e.to_string()))
    }
}
