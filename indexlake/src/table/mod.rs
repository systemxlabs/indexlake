mod alter;
mod create;
mod delete;
mod dump;
mod insert;
mod optimize;
mod scan;
mod search;
mod update;

pub use alter::*;
pub use create::*;
pub(crate) use delete::*;
pub(crate) use dump::*;
pub use insert::*;
pub use optimize::*;
pub use scan::*;
pub use search::*;
pub use update::*;

use crate::catalog::{
    Catalog, CatalogHelper, DataFileRecord, FieldRecord, INTERNAL_ROW_ID_FIELD_NAME,
    INTERNAL_ROW_ID_FIELD_REF, IndexFileRecord, Scalar, TaskRecord, TransactionHelper,
    inline_row_table_name,
};
use crate::expr::Expr;
use crate::index::{FilterSupport, IndexManager};
use crate::storage::{DataFileFormat, Storage};
use crate::utils::{
    build_row_id_array, correct_batch_schema, rewrite_batch_schema, timestamp_ms_from_now,
};
use crate::{ILError, ILResult, RecordBatchStream};
use arrow::array::{ArrayRef, RecordBatch};
use arrow::datatypes::{DataType, SchemaRef};
use arrow_schema::{Field, Schema};
use futures::StreamExt;
use log::warn;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

pub type TableSchemaRef = Arc<TableSchema>;

#[derive(Debug, Clone)]
pub struct TableSchema {
    pub arrow_schema: SchemaRef,
    pub field_name_id_map: HashMap<String, Uuid>,
    pub field_id_name_map: HashMap<Uuid, String>,
    pub field_id_default_expr_map: HashMap<Uuid, Expr>,
}

impl TableSchema {
    pub fn new(field_records: &[FieldRecord], schema_metadata: HashMap<String, String>) -> Self {
        let field_name_id_map = field_records
            .iter()
            .map(|record| (record.field_name.clone(), record.field_id))
            .collect::<HashMap<String, Uuid>>();
        let field_id_name_map = field_records
            .iter()
            .map(|record| (record.field_id, record.field_name.clone()))
            .collect::<HashMap<Uuid, String>>();
        let field_id_default_expr_map = field_records
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
        let schema = Arc::new(Schema::new_with_metadata(fields, schema_metadata));

        Self {
            arrow_schema: schema,
            field_name_id_map,
            field_id_name_map,
            field_id_default_expr_map,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Table {
    pub namespace_id: Uuid,
    pub namespace_name: String,
    pub table_id: Uuid,
    pub table_name: String,
    pub field_records: Arc<Vec<FieldRecord>>,
    pub(crate) table_schema: TableSchemaRef,
    pub output_schema: SchemaRef,
    pub config: Arc<TableConfig>,
    pub catalog: Arc<dyn Catalog>,
    pub storage: Arc<dyn Storage>,
    pub index_manager: Arc<IndexManager>,
}

impl Table {
    pub(crate) async fn transaction_helper(&self) -> ILResult<TransactionHelper> {
        TransactionHelper::new(&self.catalog).await
    }

    pub async fn create_index(self, index_creation: IndexCreation) -> ILResult<()> {
        let index_creation =
            index_creation.rewrite_columns(&self.table_schema.field_name_id_map)?;
        let mut tx_helper = self.transaction_helper().await?;
        process_create_index(&mut tx_helper, self, index_creation).await?;
        tx_helper.commit().await?;
        Ok(())
    }

    pub async fn insert(&self, insert: TableInsertion) -> ILResult<()> {
        let insert = insert.rewrite_columns(&self.table_schema.field_name_id_map)?;
        let rewritten_batches = check_and_rewrite_insert_batches(
            &insert.data,
            &self.table_schema,
            insert.ignore_row_id,
        )?;

        process_insert_into_inline_rows_without_tx(self, &rewritten_batches).await?;

        if insert.try_dump {
            try_run_dump_task(self).await?;
        }

        Ok(())
    }

    pub async fn bypass_insert(&self, stream: RecordBatchStream) -> ILResult<usize> {
        let table_schema = self.table_schema.clone();
        let stream = stream
            .map(move |batch| {
                let batch = batch?;
                let batch = rewrite_batch_schema(&batch, &table_schema.field_name_id_map)?;
                let mut batches = check_and_rewrite_insert_batches(&[batch], &table_schema, true)?;
                Ok::<_, ILError>(batches.remove(0))
            })
            .boxed();

        let count = process_bypass_insert(self, stream).await?;

        Ok(count)
    }

    pub async fn scan(&self, scan: TableScan) -> ILResult<RecordBatchStream> {
        let scan = scan.rewrite_columns(&self.table_schema.field_name_id_map)?;
        scan.validate()?;

        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let batch_stream = process_scan(&catalog_helper, self, scan).await?;

        let field_id_name_map = self.table_schema.field_id_name_map.clone();
        let correct_batch_stream = batch_stream
            .map(move |batch| {
                let batch = batch?;
                let correct_batch = correct_batch_schema(&batch, &field_id_name_map)?;
                Ok::<_, ILError>(correct_batch)
            })
            .boxed();
        Ok(correct_batch_stream)
    }

    pub async fn count(&self, partitions: &[TableScanPartition]) -> ILResult<Vec<usize>> {
        for partition in partitions {
            partition.validate()?;
        }
        if partitions.is_empty() {
            return Ok(Vec::new());
        }

        let catalog_helper = CatalogHelper::new(self.catalog.clone());

        let inline_row_count = if partitions
            .iter()
            .any(|partition| partition.contains_inline_rows())
        {
            catalog_helper
                .count_inline_rows(&self.table_id, &[])
                .await? as usize
        } else {
            0
        };

        let needs_auto_partition = partitions
            .iter()
            .any(|partition| matches!(partition, TableScanPartition::Auto { .. }));
        let mut ordered_data_files = if needs_auto_partition {
            let mut data_files = catalog_helper.get_data_files(&self.table_id).await?;
            data_files.sort_by(|left, right| {
                left.data_file_id
                    .as_bytes()
                    .cmp(right.data_file_id.as_bytes())
            });
            Some(data_files)
        } else {
            None
        };
        let data_file_count = ordered_data_files
            .as_ref()
            .map(|files| files.len())
            .unwrap_or(0);

        let mut counts = Vec::with_capacity(partitions.len());
        for partition in partitions {
            let data_file_row_count: usize = match partition {
                TableScanPartition::Provided {
                    data_file_records, ..
                } => data_file_records
                    .iter()
                    .map(|record| record.valid_row_count())
                    .sum(),
                TableScanPartition::Auto { .. } => {
                    let TableScanPartition::Auto {
                        partition_idx,
                        partition_count,
                    } = partition
                    else {
                        0
                    };
                    let (offset, limit) = TableScanPartition::data_file_offset_limit(
                        data_file_count,
                        *partition_count,
                        *partition_idx,
                    );
                    if limit == 0 || offset >= data_file_count {
                        0
                    } else {
                        let end = std::cmp::min(offset + limit, data_file_count);
                        let records = ordered_data_files.as_ref().expect("data files loaded");
                        records[offset..end]
                            .iter()
                            .map(|record| record.valid_row_count())
                            .sum()
                    }
                }
            };
            let inline_count = if partition.contains_inline_rows() {
                inline_row_count
            } else {
                0
            };
            counts.push(inline_count + data_file_row_count);
        }

        Ok(counts)
    }

    pub async fn inline_row_count(&self) -> ILResult<usize> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let count = catalog_helper
            .count_inline_rows(&self.table_id, &[])
            .await?;
        Ok(count as usize)
    }

    pub async fn search(&self, search: TableSearch) -> ILResult<RecordBatchStream> {
        let batch_stream = process_search(self, search).await?;

        let field_id_name_map = self.table_schema.field_id_name_map.clone();
        let correct_batch_stream = batch_stream
            .map(move |batch| {
                let batch = batch?;
                let correct_batch = correct_batch_schema(&batch, &field_id_name_map)?;
                Ok::<_, ILError>(correct_batch)
            })
            .boxed();
        Ok(correct_batch_stream)
    }

    pub async fn update(&self, update: TableUpdate) -> ILResult<usize> {
        let update = update.rewrite_columns(&self.table_schema.field_name_id_map)?;
        // TODO update all rows
        update
            .condition
            .check_data_type(&self.table_schema.arrow_schema, &DataType::Boolean)?;

        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let data_file_records = catalog_helper.get_data_files(&self.table_id).await?;

        let matched_data_file_rows = parallel_find_matched_data_file_rows(
            self.storage.clone(),
            self.table_schema.clone(),
            update.condition.clone(),
            data_file_records,
        )
        .await?;

        let mut tx_helper = self.transaction_helper().await?;
        let update_count =
            process_update_by_condition(&mut tx_helper, self, update, matched_data_file_rows)
                .await?;
        tx_helper.commit().await?;

        Ok(update_count)
    }

    pub async fn delete(&self, condition: Expr) -> ILResult<usize> {
        let condition = condition.rewrite_columns(&self.table_schema.field_name_id_map)?;
        condition.check_data_type(&self.table_schema.arrow_schema, &DataType::Boolean)?;

        if let Ok(scalar) = condition.constant_eval()
            && matches!(scalar, Scalar::Boolean(Some(true)))
        {
            return self.truncate().await;
        }

        // TODO waiting https://github.com/apache/arrow-rs/issues/7299 to use file row position, so we can merge below two branches
        if condition.only_visit_row_id_column() {
            let mut tx_helper = self.transaction_helper().await?;
            let delete_count =
                process_delete_by_row_id_condition(&mut tx_helper, self, &condition).await?;
            tx_helper.commit().await?;
            Ok(delete_count)
        } else {
            let catalog_helper = CatalogHelper::new(self.catalog.clone());
            let data_file_records = catalog_helper.get_data_files(&self.table_id).await?;
            let matched_data_file_row_ids = parallel_find_matched_data_file_row_ids(
                self.storage.clone(),
                self.table_schema.clone(),
                condition.clone(),
                data_file_records,
            )
            .await?;

            let mut tx_helper = self.transaction_helper().await?;
            let delete_count = process_delete_by_condition(
                &mut tx_helper,
                self,
                &condition,
                matched_data_file_row_ids,
            )
            .await?;
            tx_helper.commit().await?;
            Ok(delete_count)
        }
    }

    // Delete all rows in the table
    pub async fn truncate(&self) -> ILResult<usize> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let inline_truncate_count = catalog_helper
            .count_inline_rows(&self.table_id, &[])
            .await?;
        let data_file_records = catalog_helper.get_data_files(&self.table_id).await?;
        let index_file_records = catalog_helper.get_table_index_files(&self.table_id).await?;

        let table_name = inline_row_table_name(&self.table_id);
        self.catalog.truncate(&table_name).await?;

        let mut tx_helper = self.transaction_helper().await?;
        tx_helper.delete_all_data_files(&self.table_id).await?;
        tx_helper.delete_table_index_files(&self.table_id).await?;
        tx_helper.commit().await?;

        let file_truncate_count = data_file_records
            .iter()
            .map(|record| record.valid_row_count())
            .sum::<usize>();

        spawn_storage_data_files_clean_task(self.storage.clone(), data_file_records);
        spawn_storage_index_files_clean_task(self.storage.clone(), index_file_records);

        Ok(inline_truncate_count as usize + file_truncate_count)
    }

    // Drop the table
    pub async fn drop(self) -> ILResult<()> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let data_file_records = catalog_helper.get_data_files(&self.table_id).await?;
        let index_file_records = catalog_helper.get_table_index_files(&self.table_id).await?;

        let mut tx_helper = self.transaction_helper().await?;

        tx_helper.drop_inline_row_table(&self.table_id).await?;

        tx_helper.delete_all_data_files(&self.table_id).await?;
        tx_helper.delete_table_index_files(&self.table_id).await?;

        tx_helper.delete_table_indexes(&self.table_id).await?;
        tx_helper.delete_fields(&self.table_id).await?;
        tx_helper.delete_table(&self.table_id).await?;

        tx_helper.commit().await?;

        spawn_storage_data_files_clean_task(self.storage.clone(), data_file_records);
        spawn_storage_index_files_clean_task(self.storage.clone(), index_file_records);
        Ok(())
    }

    pub async fn drop_index(self, index_name: &str, if_exists: bool) -> ILResult<()> {
        let Some(index_def) = self.index_manager.get_index(index_name) else {
            return if if_exists {
                Ok(())
            } else {
                Err(ILError::invalid_input(format!(
                    "Index {index_name} not found"
                )))
            };
        };

        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let index_file_records = catalog_helper
            .get_index_files_by_index_id(&index_def.index_id)
            .await?;

        let mut tx_helper = self.transaction_helper().await?;

        tx_helper.delete_index(&index_def.index_id).await?;
        tx_helper
            .delete_index_files_by_index_id(&index_def.index_id)
            .await?;

        tx_helper.commit().await?;

        spawn_storage_index_files_clean_task(self.storage.clone(), index_file_records);
        Ok(())
    }

    pub async fn alter(self, alter: TableAlter) -> ILResult<()> {
        let mut tx_helper = self.transaction_helper().await?;

        process_table_alter(&mut tx_helper, &self, alter).await?;

        tx_helper.commit().await?;

        Ok(())
    }

    pub async fn optimize(&self, optimization: TableOptimization) -> ILResult<()> {
        process_table_optimization(self, optimization).await?;
        Ok(())
    }

    pub fn supports_filter(&self, filter: Expr) -> ILResult<FilterSupport> {
        let filter = filter.rewrite_columns(&self.table_schema.field_name_id_map)?;
        match filter {
            Expr::Function(_) => self.index_manager.supports_filter(&filter),
            _ => Ok(FilterSupport::Exact),
        }
    }

    pub async fn data_file_count(&self) -> ILResult<usize> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let count = catalog_helper.count_data_files(&self.table_id).await?;
        Ok(count as usize)
    }

    pub async fn data_file_records(&self) -> ILResult<Vec<DataFileRecord>> {
        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let data_file_records = catalog_helper.get_data_files(&self.table_id).await?;
        Ok(data_file_records)
    }

    pub async fn size(&self) -> ILResult<usize> {
        let inline_table_name = inline_row_table_name(&self.table_id);
        let inline_table_size = self.catalog.size(&inline_table_name).await?;

        let catalog_helper = CatalogHelper::new(self.catalog.clone());
        let data_files_size = catalog_helper.get_data_files_size(&self.table_id).await?;
        let index_files_size = catalog_helper.get_index_files_size(&self.table_id).await?;

        Ok(inline_table_size + data_files_size as usize + index_files_size as usize)
    }
}

fn spawn_storage_data_files_clean_task(
    storage: Arc<dyn Storage>,
    data_file_records: Vec<DataFileRecord>,
) {
    tokio::spawn(async move {
        for data_file_record in data_file_records {
            if let Err(e) = storage.delete(&data_file_record.relative_path).await {
                warn!(
                    "[indexlake] Failed to delete data file {}: {}",
                    data_file_record.relative_path, e
                );
            }
        }
    });
}

fn spawn_storage_index_files_clean_task(
    storage: Arc<dyn Storage>,
    index_file_records: Vec<IndexFileRecord>,
) {
    tokio::spawn(async move {
        for index_file_record in index_file_records {
            if let Err(e) = storage.delete(&index_file_record.relative_path).await {
                warn!(
                    "[indexlake] Failed to delete index file {}: {}",
                    index_file_record.relative_path, e
                );
            }
        }
    });
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableConfig {
    pub inline_row_count_limit: usize,
    pub parquet_row_group_size: usize,
    pub preferred_data_file_format: DataFileFormat,
}

impl Default for TableConfig {
    fn default() -> Self {
        Self {
            inline_row_count_limit: 100000,
            parquet_row_group_size: 1024 * 8,
            preferred_data_file_format: DataFileFormat::ParquetV2,
        }
    }
}

pub fn check_and_rewrite_insert_batches(
    batches: &[RecordBatch],
    table_schema: &TableSchemaRef,
    ignore_row_id: bool,
) -> ILResult<Vec<RecordBatch>> {
    let mut rewritten_batches = Vec::with_capacity(batches.len());
    for batch in batches {
        let batch_schema = batch.schema_ref();

        let mut fields = Vec::with_capacity(table_schema.arrow_schema.fields().len());
        let mut arrays = Vec::with_capacity(table_schema.arrow_schema.fields().len());

        for table_field in table_schema.arrow_schema.fields() {
            let internal_field_name = table_field.name();

            if internal_field_name == INTERNAL_ROW_ID_FIELD_NAME {
                fields.push(INTERNAL_ROW_ID_FIELD_REF.clone());
                if ignore_row_id {
                    let row_ids = (0..batch.num_rows())
                        .map(|_| Uuid::now_v7().into_bytes())
                        .collect::<Vec<_>>();
                    let row_id_array = build_row_id_array(row_ids.into_iter())?;
                    arrays.insert(0, Arc::new(row_id_array) as ArrayRef);
                } else {
                    let Ok(batch_field_idx) = batch_schema.index_of(INTERNAL_ROW_ID_FIELD_NAME)
                    else {
                        return Err(ILError::invalid_input(format!(
                            "Not found field {INTERNAL_ROW_ID_FIELD_NAME} in batch schema",
                        )));
                    };
                    arrays.push(batch.column(batch_field_idx).clone());
                }
            } else {
                let Ok(field_id) = Uuid::parse_str(internal_field_name) else {
                    return Err(ILError::internal(format!(
                        "Failed to parse internal field name {internal_field_name} to uuid"
                    )));
                };
                let output_field_name =
                    table_schema
                        .field_id_name_map
                        .get(&field_id)
                        .ok_or_else(|| {
                            ILError::internal(format!(
                                "Failed to find output field name for {field_id}"
                            ))
                        })?;

                if let Ok(batch_field_idx) = batch_schema.index_of(internal_field_name) {
                    let batch_field = batch_schema
                        .fields
                        .get(batch_field_idx)
                        .cloned()
                        .expect("field index should be valid");
                    check_insert_batch_field(&batch_field, table_field, output_field_name)?;

                    fields.push(batch_field);
                    arrays.push(batch.column(batch_field_idx).clone());
                } else if let Some(default_expr) =
                    table_schema.field_id_default_expr_map.get(&field_id)
                {
                    fields.push(table_field.clone());
                    let array = eval_default_expr(default_expr, batch, table_field.as_ref())?;
                    arrays.push(array);
                } else {
                    return Err(ILError::invalid_input(format!(
                        "Missing field {output_field_name} (no default expr) in record batch",
                    )));
                }
            }
        }

        let rewritten_batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)?;
        rewritten_batches.push(rewritten_batch);
    }

    Ok(rewritten_batches)
}

pub(crate) fn eval_default_expr(
    expr: &Expr,
    batch: &RecordBatch,
    field: &Field,
) -> ILResult<ArrayRef> {
    let value = expr.eval(batch)?;
    let array = value.into_array(batch.num_rows())?;
    if array.data_type() != field.data_type() {
        return Err(ILError::invalid_input(format!(
            "Default expr type {} does not match field {}",
            array.data_type(),
            field
        )));
    }
    if !field.is_nullable() && array.null_count() > 0 {
        return Err(ILError::invalid_input(format!(
            "Default expr produced null for non-nullable field {}",
            field
        )));
    }
    Ok(array)
}

pub(crate) fn check_default_expr(field: &Field, expr: &Expr, schema: &Schema) -> ILResult<()> {
    if contains_unsupported_default_expr(expr) {
        return Err(ILError::invalid_input(
            "Default expr contains unsupported operator".to_string(),
        ));
    }

    let expr_type = expr.data_type(schema)?;
    if &expr_type != field.data_type() {
        return Err(ILError::invalid_input(format!(
            "Default expr data type {expr_type} does not match field {field}",
        )));
    }

    if let Ok(constant) = expr.constant_eval()
        && constant.is_null()
        && !field.is_nullable()
    {
        return Err(ILError::invalid_input(format!(
            "Default expr is null for non-nullable field {field}",
        )));
    }

    Ok(())
}

fn contains_unsupported_default_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Function(_) | Expr::TryCast(_) => true,
        Expr::Column(_) | Expr::Literal(_) => false,
        Expr::BinaryExpr(binary_expr) => {
            contains_unsupported_default_expr(&binary_expr.left)
                || contains_unsupported_default_expr(&binary_expr.right)
        }
        Expr::Not(inner) | Expr::IsNull(inner) | Expr::IsNotNull(inner) | Expr::Negative(inner) => {
            contains_unsupported_default_expr(inner)
        }
        Expr::InList(in_list) => {
            contains_unsupported_default_expr(&in_list.expr)
                || in_list.list.iter().any(contains_unsupported_default_expr)
        }
        Expr::Like(like) => {
            contains_unsupported_default_expr(&like.expr)
                || contains_unsupported_default_expr(&like.pattern)
        }
        Expr::Cast(cast) => contains_unsupported_default_expr(&cast.expr),
        Expr::Case(case) => {
            case.when_then.iter().any(|(when, then)| {
                contains_unsupported_default_expr(when) || contains_unsupported_default_expr(then)
            }) || case
                .else_expr
                .as_ref()
                .map(|expr| contains_unsupported_default_expr(expr))
                .unwrap_or(false)
        }
    }
}

pub fn check_insert_batch_field(
    batch_field: &Field,
    table_field: &Field,
    output_field_name: &str,
) -> ILResult<()> {
    if batch_field.name() != table_field.name()
        || batch_field.data_type() != table_field.data_type()
        || batch_field.is_nullable() != table_field.is_nullable()
    {
        return Err(ILError::invalid_input(format!(
            "Invalid batch field of name {output_field_name}: {batch_field:?}, expected field: {table_field:?}",
        )));
    }

    Ok(())
}

pub(crate) async fn insert_task(
    catalog: &Arc<dyn Catalog>,
    task_id: String,
    max_lifetime: Duration,
) -> ILResult<()> {
    let mut tx_helper = TransactionHelper::new(catalog).await?;
    tx_helper.delete_expired_tasks().await?;
    let task = TaskRecord {
        task_id,
        start_at: timestamp_ms_from_now(Duration::ZERO),
        max_lifetime: max_lifetime.as_millis() as i64,
    };
    tx_helper.insert_task(task).await?;
    tx_helper.commit().await?;
    Ok(())
}
