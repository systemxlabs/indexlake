use std::collections::HashMap;
use std::sync::Arc;

use uuid::Uuid;

use crate::expr::Expr;
use crate::index::{FilterSupport, IndexBuilder, IndexDefinitionRef, IndexKind};
use crate::{ILError, ILResult};

#[derive(Debug, Clone)]
pub struct IndexManager {
    indexes: Vec<IndexDefinitionRef>,
    kinds: HashMap<String, Arc<dyn IndexKind>>,
}

impl IndexManager {
    pub(crate) fn try_new(
        indexes: Vec<IndexDefinitionRef>,
        kinds: HashMap<String, Arc<dyn IndexKind>>,
    ) -> ILResult<Self> {
        for index_def in &indexes {
            if !kinds.contains_key(&index_def.kind) {
                return Err(ILError::invalid_input(format!(
                    "Index kind {} not registered",
                    index_def.kind
                )));
            }
        }
        Ok(Self { indexes, kinds })
    }

    pub(crate) fn get_index(&self, index_name: &str) -> Option<&IndexDefinitionRef> {
        self.indexes.iter().find(|index| index.name == index_name)
    }

    pub(crate) fn get_index_kind(&self, kind: &str) -> Option<&Arc<dyn IndexKind>> {
        self.kinds.get(kind)
    }

    pub(crate) fn any_index_contains_field(&self, field_id: &Uuid) -> bool {
        let field_id_str = hex::encode(field_id);
        self.indexes
            .iter()
            .any(|def| def.key_columns.contains(&field_id_str))
    }

    pub(crate) fn supports_filter(&self, filter: &Expr) -> ILResult<FilterSupport> {
        let mut supports = FilterSupport::Unsupported;
        for index_def in &self.indexes {
            let index_kind = self
                .kinds
                .get(&index_def.kind)
                .expect("Index kind not found");
            match index_kind.supports_filter(index_def, filter)? {
                FilterSupport::Exact => return Ok(FilterSupport::Exact),
                FilterSupport::Inexact => supports = FilterSupport::Inexact,
                FilterSupport::Unsupported => (),
            }
        }
        Ok(supports)
    }

    pub(crate) fn iter_index_and_kind(
        &self,
    ) -> impl Iterator<Item = (&IndexDefinitionRef, &Arc<dyn IndexKind>)> {
        self.indexes.iter().map(|index_def| {
            let index_kind = self
                .kinds
                .get(&index_def.kind)
                .expect("Index kind not found");
            (index_def, index_kind)
        })
    }

    pub(crate) fn new_index_builders(&self) -> ILResult<Vec<Box<dyn IndexBuilder>>> {
        self.indexes
            .iter()
            .map(|index_def| {
                let index_kind = self
                    .kinds
                    .get(&index_def.kind)
                    .expect("Index kind not found");
                index_kind.builder(index_def)
            })
            .collect()
    }

    pub(crate) fn new_mergeable_index_builders(&self) -> ILResult<Vec<Box<dyn IndexBuilder>>> {
        let mut builders = Vec::new();
        for index_def in &self.indexes {
            let index_kind = self
                .kinds
                .get(&index_def.kind)
                .expect("Index kind not found");
            let builder = index_kind.builder(index_def)?;
            if builder.mergeable() {
                builders.push(builder);
            }
        }
        Ok(builders)
    }

    pub(crate) fn new_non_mergeable_index_builders(&self) -> ILResult<Vec<Box<dyn IndexBuilder>>> {
        let mut builders = Vec::new();
        for index_def in &self.indexes {
            let index_kind = self
                .kinds
                .get(&index_def.kind)
                .expect("Index kind not found");
            let builder = index_kind.builder(index_def)?;
            if !builder.mergeable() {
                builders.push(builder);
            }
        }
        Ok(builders)
    }
}
