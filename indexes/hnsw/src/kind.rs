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

use crate::{HnswIndexBuilder, HnswSearchQuery};

#[derive(Debug)]
pub struct HnswIndexKind;

impl IndexKind for HnswIndexKind {
    fn kind(&self) -> &str {
        "hnsw"
    }

    fn decode_params(&self, value: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params: HnswIndexParams = serde_json::from_str(value)
            .map_err(|e| ILError::index(format!("Failed to decode HnswIndexParams: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::index("Hnsw index requires exactly one key column"));
        }
        let key_column_name = &index_def.key_columns[0];
        let key_field = index_def
            .table_schema
            .arrow_schema
            .field_with_name(key_column_name)?;
        match key_field.data_type() {
            DataType::List(inner) => {
                if !matches!(inner.data_type(), DataType::Float32) || inner.is_nullable() {
                    return Err(ILError::index(
                        "Hnsw index key column must be a list of non-nullable float32",
                    ));
                }
            }
            _ => {
                return Err(ILError::index(
                    "Hnsw index key column must be a list of non-nullable float32",
                ));
            }
        }
        Ok(())
    }

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(HnswIndexBuilder::try_new(index_def.clone())?))
    }

    fn supports_search(
        &self,
        _index_def: &IndexDefinition,
        query: &dyn SearchQuery,
    ) -> ILResult<bool> {
        let Some(_query) = query.as_any().downcast_ref::<HnswSearchQuery>() else {
            return Ok(false);
        };
        Ok(true)
    }

    fn supports_filter(
        &self,
        _index_def: &IndexDefinition,
        _filter: &Expr,
    ) -> ILResult<FilterSupport> {
        Ok(FilterSupport::Unsupported)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswIndexParams {
    pub ef_construction: usize,
}

impl IndexParams for HnswIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self) -> ILResult<String> {
        let json = serde_json::to_string(self)
            .map_err(|e| ILError::index(format!("Failed to encode HnswIndexParams: {e}")))?;
        Ok(json)
    }
}
