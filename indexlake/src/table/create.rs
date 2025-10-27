use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow_schema::Schema;
use futures::StreamExt;
use uuid::Uuid;

use crate::catalog::{
    CatalogSchema, FieldRecord, IndexFileRecord, IndexRecord, InlineIndexRecord, Scalar,
    TableRecord, TransactionHelper, rows_to_record_batch,
};
use crate::index::{IndexDefinition, IndexParams};
use crate::storage::read_data_file_by_record;
use crate::table::{Table, TableConfig};
use crate::{ILError, ILResult};

#[derive(Debug, Clone)]
pub struct TableCreation {
    pub namespace_name: String,
    pub table_name: String,
    pub schema: SchemaRef,
    pub default_values: HashMap<String, Scalar>,
    pub config: TableConfig,
    pub if_not_exists: bool,
}

impl Default for TableCreation {
    fn default() -> Self {
        Self {
            namespace_name: String::default(),
            table_name: String::default(),
            schema: Arc::new(Schema::empty()),
            default_values: HashMap::default(),
            config: TableConfig::default(),
            if_not_exists: false,
        }
    }
}

pub(crate) async fn process_create_table(
    tx_helper: &mut TransactionHelper,
    creation: TableCreation,
) -> ILResult<Uuid> {
    let namespace_id = tx_helper
        .get_namespace_id(&creation.namespace_name)
        .await?
        .ok_or_else(|| {
            ILError::invalid_input(format!("Namespace {} not found", creation.namespace_name))
        })?;

    if let Some(table_id) = tx_helper
        .get_table_id(&namespace_id, &creation.table_name)
        .await?
    {
        return if creation.if_not_exists {
            Ok(table_id)
        } else {
            Err(ILError::invalid_input(format!(
                "Table {} already exists in namespace {}",
                creation.table_name, creation.namespace_name
            )))
        };
    }

    // check default values
    for (field_name, default_value) in &creation.default_values {
        let default_value_type = default_value.data_type();
        let field = creation.schema.field_with_name(field_name)?;
        if &default_value_type != field.data_type() {
            return Err(ILError::invalid_input(format!(
                "Default value data type {default_value_type} does not match field {field}",
            )));
        }
        if default_value.is_null() && !field.is_nullable() {
            return Err(ILError::invalid_input(format!(
                "Default value is null for non-nullable field {field}",
            )));
        }
    }

    // insert table record
    let table_id = Uuid::now_v7();
    tx_helper
        .insert_table(&TableRecord {
            table_id,
            table_name: creation.table_name,
            namespace_id,
            config: creation.config,
            schema_metadata: creation.schema.metadata().clone(),
        })
        .await?;

    // insert field records
    let mut field_records = Vec::new();
    for field in creation.schema.fields() {
        let default_value = creation.default_values.get(field.name()).cloned();
        field_records.push(FieldRecord::new(
            Uuid::now_v7(),
            table_id,
            field,
            default_value,
        ));
    }
    tx_helper.insert_fields(&field_records).await?;

    tx_helper
        .create_inline_row_table(&table_id, &field_records)
        .await?;

    Ok(table_id)
}

#[derive(Debug, Clone)]
pub struct IndexCreation {
    pub name: String,
    pub kind: String,
    pub key_columns: Vec<String>,
    pub params: Arc<dyn IndexParams>,
    pub if_not_exists: bool,
}

impl IndexCreation {
    pub(crate) fn rewrite_columns(
        mut self,
        field_name_id_map: &HashMap<String, Uuid>,
    ) -> ILResult<Self> {
        let mut rewritten_columns = Vec::with_capacity(self.key_columns.len());
        for key_col in self.key_columns {
            if let Some(field_id) = field_name_id_map.get(&key_col) {
                rewritten_columns.push(hex::encode(field_id));
            } else {
                return Err(ILError::invalid_input(format!(
                    "key column {key_col} not found"
                )));
            }
        }
        self.key_columns = rewritten_columns;
        Ok(self)
    }
}

pub(crate) async fn process_create_index(
    tx_helper: &mut TransactionHelper,
    table: Table,
    creation: IndexCreation,
) -> ILResult<Uuid> {
    let index_id = Uuid::now_v7();
    let index_def = Arc::new(IndexDefinition {
        index_id,
        name: creation.name.clone(),
        kind: creation.kind.clone(),
        table_id: table.table_id,
        table_name: table.table_name.clone(),
        table_schema: table.schema.clone(),
        key_columns: creation.key_columns.clone(),
        params: creation.params.clone(),
    });

    let index_kind = table
        .index_manager
        .get_index_kind(&creation.kind)
        .ok_or_else(|| {
            ILError::invalid_input(format!("Index kind {} not registered", creation.kind))
        })?;
    index_kind.supports(&index_def)?;

    if let Some(index_id) = tx_helper
        .get_index_id(&table.table_id, &creation.name)
        .await?
    {
        return if creation.if_not_exists {
            Ok(index_id)
        } else {
            Err(ILError::invalid_input(format!(
                "Index name {} already exists in table {}",
                creation.name, table.table_name
            )))
        };
    }

    // create inline index
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&table.schema)?);
    let row_stream = tx_helper
        .scan_inline_rows(&table.table_id, &catalog_schema, &[], None, None)
        .await?;
    let table_schema = table.schema.clone();
    let mut inline_stream = row_stream.chunks(100).map(move |rows| {
        let rows = rows.into_iter().collect::<ILResult<Vec<_>>>()?;
        let batch = rows_to_record_batch(&table_schema, &rows)?;
        Ok::<_, ILError>(batch)
    });
    let mut index_builder = index_kind.builder(&index_def)?;
    while let Some(batch) = inline_stream.next().await {
        let batch = batch?;
        index_builder.append(&batch)?;
    }
    drop(inline_stream);
    let mut index_data = Vec::new();
    index_builder.write_bytes(&mut index_data)?;
    tx_helper
        .insert_inline_indexes(&[InlineIndexRecord {
            index_id,
            index_data,
        }])
        .await?;

    // create index file
    let mut projection = vec![0];
    for col in creation.key_columns.iter() {
        let idx = table.schema.index_of(col)?;
        projection.push(idx);
    }
    projection.sort();

    let data_file_records = tx_helper.get_data_files(&table.table_id).await?;

    let mut index_file_records = Vec::new();
    for data_file_record in data_file_records {
        let mut index_builder = index_kind.builder(&index_def)?;
        let mut stream = read_data_file_by_record(
            table.storage.as_ref(),
            &table.schema,
            &data_file_record,
            Some(projection.clone()),
            vec![],
            None,
            1024,
        )
        .await?;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            index_builder.append(&batch)?;
        }

        let index_file_id = Uuid::now_v7();
        let relative_path = IndexFileRecord::build_relative_path(
            &table.namespace_id,
            &table.table_id,
            &index_file_id,
        );
        let output_file = table.storage.create(&relative_path).await?;
        index_builder.write_file(output_file).await?;
        index_file_records.push(IndexFileRecord {
            index_file_id,
            table_id: table.table_id,
            index_id,
            data_file_id: data_file_record.data_file_id,
            relative_path,
        });
    }

    tx_helper.insert_index_files(&index_file_records).await?;

    let key_field_ids = creation
        .key_columns
        .iter()
        .map(|c| Uuid::parse_str(c))
        .collect::<Result<Vec<Uuid>, _>>()?;

    tx_helper
        .insert_index(&IndexRecord {
            index_id,
            index_name: creation.name.clone(),
            index_kind: creation.kind.clone(),
            table_id: table.table_id,
            key_field_ids,
            params: creation.params.encode()?,
        })
        .await?;

    Ok(index_id)
}
