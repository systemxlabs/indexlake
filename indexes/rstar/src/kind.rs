use std::sync::Arc;

use arrow::datatypes::DataType;
use serde::{Deserialize, Serialize};

use geozero::wkb::WkbDialect as GeozeroWkbDialect;
use indexlake::catalog::Scalar;
use indexlake::expr::{Expr, Function};
use indexlake::index::{
    FilterSupport, IndexBuilder, IndexDefinition, IndexDefinitionRef, IndexKind, IndexParams,
    SearchQuery,
};
use indexlake::{ILError, ILResult};

use crate::RStarIndexBuilder;

#[derive(Debug)]
pub struct RStarIndexKind;

impl IndexKind for RStarIndexKind {
    fn kind(&self) -> &str {
        "rstar"
    }

    fn decode_params(&self, value: &str) -> ILResult<Arc<dyn IndexParams>> {
        let params = serde_json::from_str::<RStarIndexParams>(value)
            .map_err(|e| ILError::index(format!("Failed to parse RStarIndexParams: {e}")))?;
        Ok(Arc::new(params))
    }

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()> {
        if index_def.key_columns.len() != 1 {
            return Err(ILError::index(
                "RStar index requires exactly one key column",
            ));
        }
        let key_column_name = &index_def.key_columns[0];
        let key_field = index_def
            .table_schema
            .arrow_schema
            .field_with_name(key_column_name)?;
        if !matches!(
            key_field.data_type(),
            DataType::Binary | DataType::LargeBinary | DataType::BinaryView
        ) {
            return Err(ILError::index(
                "RStar index key column must be a binary / large binary / binary view column",
            ));
        }
        Ok(())
    }

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>> {
        Ok(Box::new(RStarIndexBuilder::try_new(index_def.clone())?))
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
            Expr::Function(Function {
                name,
                args,
                return_type,
            }) => match name.to_ascii_lowercase().as_str() {
                "intersects" | "st_intersects" => {
                    check_intersects_function(index_def, args, return_type)
                }
                _ => Ok(FilterSupport::Unsupported),
            },
            _ => Ok(FilterSupport::Unsupported),
        }
    }
}

fn check_intersects_function(
    index_def: &IndexDefinition,
    args: &[Expr],
    _return_type: &DataType,
) -> ILResult<FilterSupport> {
    if args.len() != 2 {
        return Ok(FilterSupport::Unsupported);
    }

    let arg0 = &args[0];
    let Expr::Column(col) = arg0 else {
        return Ok(FilterSupport::Unsupported);
    };
    let key_column_name = &index_def.key_columns[0];
    let key_field = index_def
        .table_schema
        .arrow_schema
        .field_with_name(key_column_name)?;
    if key_field.name() != col {
        return Ok(FilterSupport::Unsupported);
    }

    let arg1 = &args[1];
    let Expr::Literal(literal) = arg1 else {
        return Ok(FilterSupport::Unsupported);
    };
    if !matches!(literal.value, Scalar::Binary(Some(_))) {
        return Ok(FilterSupport::Unsupported);
    }
    Ok(FilterSupport::Inexact)
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
            .map_err(|e| ILError::index(format!("Failed to serialize RStarIndexParams: {e}")))
    }
}
