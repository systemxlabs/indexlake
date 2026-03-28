use std::any::Any;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, FieldRef};
use indexlake::expr::Expr;
use indexlake::index::{
    FilterSupport, IndexBuilder, IndexDefinition, IndexDefinitionRef, IndexKind, IndexParams,
    SearchQuery,
};
use indexlake::{ILError, ILResult};
use serde::{Deserialize, Serialize};

use crate::{RabitqIndexBuilder, RabitqSearchQuery};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RabitqAlgo {
    BruteForce,
    Ivf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RabitqMetric {
    L2,
    InnerProduct,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RabitqIndexParams {
    pub algo: RabitqAlgo,
    pub metric: RabitqMetric,
    /// Number of bits for quantization.
    #[serde(default = "default_total_bits")]
    pub total_bits: usize,
    /// IVF: number of clusters (nlist).
    #[serde(default = "default_nlist")]
    pub nlist: usize,
}

fn default_total_bits() -> usize {
    7
}

fn default_nlist() -> usize {
    256
}

impl IndexParams for RabitqIndexParams {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn encode(&self) -> ILResult<String> {
        serde_json::to_string(self)
            .map_err(|e| ILError::index(format!("Failed to encode RabitqIndexParams: {e}")))
    }
}

#[derive(Debug)]
pub struct RabitqIndexKind;

impl IndexKind for RabitqIndexKind {
    fn kind(&self) -> &str {
        "rabitq"
    }

    fn decode_params(&self, value: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params: RabitqIndexParams = serde_json::from_str(value)
            .map_err(|e| ILError::index(format!("Failed to decode RabitqIndexParams: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::index(
                "RaBitQ index requires exactly one key column",
            ));
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
                        "RaBitQ index key column must be a list of non-nullable float32",
                    ));
                }
            }
            _ => {
                return Err(ILError::index(
                    "RaBitQ index key column must be a list of non-nullable float32",
                ));
            }
        }
        Ok(())
    }

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(RabitqIndexBuilder::try_new(index_def.clone())?))
    }

    fn supports_search(
        &self,
        _index_def: &IndexDefinition,
        query: &dyn SearchQuery,
    ) -> ILResult<bool> {
        Ok(query.as_any().downcast_ref::<RabitqSearchQuery>().is_some())
    }

    fn dynamic_fields(&self, _index_def: &IndexDefinition) -> ILResult<Vec<FieldRef>> {
        Ok(vec![Arc::new(Field::new(
            "score",
            DataType::Float64,
            false,
        ))])
    }

    fn supports_filter(
        &self,
        _index_def: &IndexDefinition,
        _filter: &Expr,
    ) -> ILResult<FilterSupport> {
        Ok(FilterSupport::Unsupported)
    }
}
