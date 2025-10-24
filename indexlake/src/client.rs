use arrow::datatypes::Schema;
use arrow_schema::Field;
use uuid::Uuid;

use crate::catalog::{
    Catalog, CatalogHelper, INTERNAL_ROW_ID_FIELD_NAME, INTERNAL_ROW_ID_FIELD_REF,
    TransactionHelper,
};
use crate::index::{IndexDefinition, IndexKind, IndexManager};
use crate::storage::Storage;
use crate::table::{MetadataColumn, Table, TableCreation, process_create_table};
use crate::{ILError, ILResult};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Client {
    pub catalog: Arc<dyn Catalog>,
    pub storage: Arc<Storage>,
    pub index_kinds: HashMap<String, Arc<dyn IndexKind>>,
}

impl Client {
    pub fn new(catalog: Arc<dyn Catalog>, storage: Arc<Storage>) -> Self {
        Self {
            catalog,
            storage,
            index_kinds: HashMap::new(),
        }
    }

    pub(crate) async fn transaction_helper(&self) -> ILResult<TransactionHelper> {
        TransactionHelper::new(&self.catalog).await
    }

    pub fn register_index_kind(&mut self, index_kind: Arc<dyn IndexKind>) {
        self.index_kinds
            .insert(index_kind.kind().to_string(), index_kind);
    }

    pub async fn create_namespace(
        &self,
        namespace_name: &str,
        if_not_exists: bool,
    ) -> ILResult<Uuid> {
        let mut tx_helper = self.transaction_helper().await?;

        if let Some(namespace_id) = tx_helper.get_namespace_id(namespace_name).await? {
            if if_not_exists {
                return Ok(namespace_id);
            } else {
                return Err(ILError::invalid_input(format!(
                    "Namespace {namespace_name} already exists"
                )));
            }
        }

        let namespace_id = Uuid::now_v7();
        tx_helper
            .insert_namespace(&namespace_id, namespace_name)
            .await?;
        tx_helper.commit().await?;

        Ok(namespace_id)
    }

    pub async fn get_namespace_id(&self, namespace_name: &str) -> ILResult<Option<Uuid>> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let namespace_id = catalog_helper.get_namespace_id(namespace_name).await?;
        Ok(namespace_id)
    }

    pub async fn create_table(&self, table_creation: TableCreation) -> ILResult<Uuid> {
        if table_creation.schema.fields().is_empty() {
            return Err(ILError::invalid_input("Schema is empty"));
        }
        check_schema_contains_system_column(&table_creation.schema)?;

        let mut tx_helper = self.transaction_helper().await?;
        let table_id = process_create_table(&mut tx_helper, table_creation.clone()).await?;
        tx_helper.commit().await?;

        Ok(table_id)
    }

    pub async fn load_table(&self, namespace_name: &str, table_name: &str) -> ILResult<Table> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());

        let namespace_id = catalog_helper
            .get_namespace_id(namespace_name)
            .await?
            .ok_or_else(|| {
                ILError::invalid_input(format!("Namespace {namespace_name} not found"))
            })?;

        let table_record = catalog_helper
            .get_table(&namespace_id, table_name)
            .await?
            .ok_or_else(|| {
                ILError::invalid_input(format!(
                    "Table {table_name} not found in namespace {namespace_name}"
                ))
            })?;

        let field_records = Arc::new(
            catalog_helper
                .get_table_fields(&table_record.table_id)
                .await?,
        );
        let field_name_id_map = field_records
            .iter()
            .map(|record| (record.field_name.clone(), record.field_id))
            .collect::<HashMap<String, Uuid>>();
        let field_id_name_map = field_records
            .iter()
            .map(|record| (record.field_id, record.field_name.clone()))
            .collect::<HashMap<Uuid, String>>();
        let field_id_default_value_map = field_records
            .iter()
            .filter(|record| record.default_value.is_some())
            .map(|record| (record.field_id, record.default_value.clone().unwrap()))
            .collect::<HashMap<_, _>>();

        let mut fields = field_records
            .iter()
            .map(|f| {
                Arc::new(
                    Field::new(hex::encode(f.field_id), f.data_type.clone(), f.nullable)
                        .with_metadata(f.metadata.clone()),
                )
            })
            .collect::<Vec<_>>();
        fields.insert(0, INTERNAL_ROW_ID_FIELD_REF.clone());
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            table_record.schema_metadata.clone(),
        ));

        let mut output_fields = field_records
            .iter()
            .map(|f| {
                Arc::new(
                    Field::new(&f.field_name, f.data_type.clone(), f.nullable)
                        .with_metadata(f.metadata.clone()),
                )
            })
            .collect::<Vec<_>>();
        output_fields.insert(0, INTERNAL_ROW_ID_FIELD_REF.clone());
        let output_schema = Arc::new(Schema::new_with_metadata(
            output_fields,
            table_record.schema_metadata,
        ));

        let index_records = catalog_helper
            .get_table_indexes(&table_record.table_id)
            .await?;
        let mut indexes = Vec::new();
        for index_record in index_records {
            let index = IndexDefinition::from_index_record(
                &index_record,
                table_name,
                &schema,
                &self.index_kinds,
            )?;
            indexes.push(Arc::new(index));
        }

        let index_manager = IndexManager::try_new(indexes, self.index_kinds.clone())?;

        Ok(Table {
            namespace_id,
            namespace_name: namespace_name.to_string(),
            table_id: table_record.table_id,
            table_name: table_name.to_string(),
            field_records,
            schema,
            output_schema,
            field_name_id_map,
            field_id_name_map,
            field_id_default_value_map,
            config: Arc::new(table_record.config),
            catalog: self.catalog.clone(),
            storage: self.storage.clone(),
            index_manager: Arc::new(index_manager),
        })
    }
}

fn check_schema_contains_system_column(schema: &Schema) -> ILResult<()> {
    if schema.field_with_name(INTERNAL_ROW_ID_FIELD_NAME).is_ok() {
        return Err(ILError::invalid_input(format!(
            "Schema contains system column: {INTERNAL_ROW_ID_FIELD_NAME}"
        )));
    }

    let location_kind_field = MetadataColumn::LocationKind.to_field();
    let location_kind_field_name = location_kind_field.name();
    if schema.field_with_name(location_kind_field_name).is_ok() {
        return Err(ILError::invalid_input(format!(
            "Schema contains system column: {location_kind_field_name}"
        )));
    }

    let location_field = MetadataColumn::Location.to_field();
    let location_field_name = location_field.name();
    if schema.field_with_name(location_field_name).is_ok() {
        return Err(ILError::invalid_input(format!(
            "Schema contains system column: {location_field_name}"
        )));
    }

    Ok(())
}
