use crate::catalog::IndexRecord;
use crate::index::{IndexKind, IndexParams};
use crate::table::TableSchemaRef;
use crate::{ILError, ILResult};
use arrow::datatypes::FieldRef;
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

pub type IndexDefinitionRef = Arc<IndexDefinition>;

#[derive(Debug, Clone)]
pub struct IndexDefinition {
    pub index_id: Uuid,
    pub name: String,
    pub kind: String,
    pub table_id: Uuid,
    pub table_name: String,
    pub table_schema: TableSchemaRef,
    pub key_columns: Vec<String>,
    pub params: Arc<dyn IndexParams>,
}

impl IndexDefinition {
    pub fn key_fields(&self) -> ILResult<Vec<FieldRef>> {
        let mut key_fields = Vec::new();
        for name in self.key_columns.iter() {
            let index = self.table_schema.arrow_schema.index_of(name)?;
            let field = self
                .table_schema
                .arrow_schema
                .fields
                .get(index)
                .cloned()
                .ok_or_else(|| {
                    ILError::internal(format!(
                        "Key field {} not found in table schema {}",
                        name, self.table_schema.arrow_schema
                    ))
                })?;
            key_fields.push(field);
        }
        Ok(key_fields)
    }

    pub fn downcast_params<T: 'static>(&self) -> ILResult<&T> {
        self.params.as_any().downcast_ref::<T>().ok_or_else(|| {
            ILError::internal(format!(
                "Index params is not {}",
                std::any::type_name::<T>()
            ))
        })
    }

    pub(crate) fn from_index_record(
        index_record: &IndexRecord,
        table_name: &str,
        table_schema: &TableSchemaRef,
        index_kinds: &HashMap<String, Arc<dyn IndexKind>>,
    ) -> ILResult<Self> {
        let mut key_columns = Vec::new();
        for key_field_id in index_record.key_field_ids.iter() {
            key_columns.push(hex::encode(key_field_id));
        }

        let index_kind = index_kinds.get(&index_record.index_kind).ok_or_else(|| {
            ILError::internal(format!("Index kind {} not found", index_record.index_kind))
        })?;
        let params = index_kind.decode_params(&index_record.params)?;

        Ok(Self {
            index_id: index_record.index_id,
            name: index_record.index_name.clone(),
            kind: index_record.index_kind.clone(),
            table_id: index_record.table_id,
            table_name: table_name.to_string(),
            table_schema: table_schema.clone(),
            key_columns,
            params,
        })
    }
}
