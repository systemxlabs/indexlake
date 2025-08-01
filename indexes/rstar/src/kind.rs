use std::sync::Arc;

use arrow::datatypes::DataType;
use serde::{Deserialize, Serialize};

use geozero::wkb::WkbDialect as GeozeroWkbDialect;
use indexlake::{
    ILError, ILResult,
    catalog::Scalar,
    expr::{Expr, Function},
    index::{
        IndexBuilder, IndexDefination, IndexDefinationRef, IndexKind, IndexParams, SearchQuery,
    },
};

use crate::RStarIndexBuilder;

#[derive(Debug)]
pub struct RStarIndexKind;

impl IndexKind for RStarIndexKind {
    fn kind(&self) -> &str {
        "rstar"
    }

    fn decode_params(&self, value: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params = serde_json::from_str::<RStarIndexParams>(value)
            .map_err(|e| ILError::IndexError(format!("Failed to parse RStarIndexParams: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefination) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::IndexError(format!(
                "RStar index requires exactly one key column"
            )));
        }
        let key_column_name = &index_def.key_columns[0];
        let key_field = index_def.table_schema.field_with_name(&key_column_name)?;
        if !matches!(
            key_field.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        ) {
            return Err(ILError::IndexError(format!(
                "RStar index key column must be a binary / large binary / binary view column"
            )));
        }
        Ok(())
    }

    fn builder(&self, index_def: &IndexDefinationRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(RStarIndexBuilder::try_new(index_def.clone())?))
    }

    fn supports_search(
        &self,
        _index_def: &IndexDefination,
        _query: &dyn SearchQuery,
    ) -> ILResult<bool> {
        Ok(false)
    }

    fn supports_filter(&self, index_def: &IndexDefination, filter: &Expr) -> ILResult<bool> {
        match filter {
            Expr::Function(Function {
                name,
                args,
                return_type,
            }) => match name.as_str() {
                "intersects" => check_intersects_function(index_def, args, return_type),
                _ => Ok(false),
            },
            _ => Ok(false),
        }
    }
}

fn check_intersects_function(
    index_def: &IndexDefination,
    args: &[Expr],
    return_type: &DataType,
) -> ILResult<bool> {
    if args.len() != 2 {
        return Err(ILError::IndexError(format!(
            "Intersects function must have two arguments"
        )));
    }

    let arg0 = &args[0];
    let Expr::Column(col) = arg0 else {
        return Err(ILError::IndexError(format!(
            "Intersects function must have a column as the first argument"
        )));
    };
    let key_column_name = &index_def.key_columns[0];
    let key_field = index_def.table_schema.field_with_name(&key_column_name)?;
    if key_field.name() != col {
        return Err(ILError::IndexError(format!(
            "Intersects function must have a column with the same name as the key column"
        )));
    }

    let arg1 = &args[1];
    let Expr::Literal(scalar) = arg1 else {
        return Err(ILError::IndexError(format!(
            "Intersects function must have a literal binary as the second argument"
        )));
    };
    if !matches!(scalar, Scalar::Binary(Some(_))) {
        return Err(ILError::IndexError(format!(
            "Intersects function must have a literal binary as the second argument"
        )));
    }
    Ok(true)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RStarIndexParams {
    pub wkb_dialect: WkbDialect,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WkbDialect {
    Wkb,
    Ewkb,
    Geopackage,
    MySQL,
    SpatiaLite,
}

impl WkbDialect {
    pub fn to_geozero(&self) -> GeozeroWkbDialect {
        match self {
            WkbDialect::Wkb => GeozeroWkbDialect::Wkb,
            WkbDialect::Ewkb => GeozeroWkbDialect::Ewkb,
            WkbDialect::Geopackage => GeozeroWkbDialect::Geopackage,
            WkbDialect::MySQL => GeozeroWkbDialect::MySQL,
            WkbDialect::SpatiaLite => GeozeroWkbDialect::SpatiaLite,
        }
    }
}

impl IndexParams for RStarIndexParams {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn encode(&self) -> ILResult<String> {
        serde_json::to_string(self)
            .map_err(|e| ILError::IndexError(format!("Failed to serialize RStarIndexParams: {e}")))
    }
}
