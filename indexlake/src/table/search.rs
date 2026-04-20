use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeBinaryArray, Float64Array, RecordBatch};
use arrow::compute::SortOptions;
use arrow::datatypes::{FieldRef, Schema, SchemaRef};
use futures::TryStreamExt;
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, CatalogSchema, DataFileRecord, INTERNAL_ROW_ID_FIELD_NAME, IndexFileRecord,
    InlineIndexRecord, Row, rows_to_record_batch,
};
use crate::expr::row_ids_in_list_expr;
use crate::index::{DynamicColumn, IndexDefinitionRef, IndexKind, SearchIndexEntries, SearchQuery};
use crate::storage::{Storage, read_data_file_by_record};
use crate::table::Table;
use crate::utils::{extract_row_ids_from_record_batch, project_schema};
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, Clone)]
pub struct TableSearch {
    pub query: Arc<dyn SearchQuery>,
    pub projection: Option<Vec<usize>>,
    pub dynamic_fields: Vec<String>,
}

impl TableSearch {
    pub fn output_schema(&self, table: &Table) -> ILResult<SchemaRef> {
        let projected_schema = project_schema(&table.output_schema, self.projection.as_ref())?;
        if self.dynamic_fields.is_empty() {
            return Ok(Arc::new(projected_schema));
        }

        let index_kind_name = self.query.index_kind();
        let (index_def, index_kind) = table
            .index_manager
            .iter_index_and_kind()
            .find(|(index_def, _)| index_def.kind == index_kind_name)
            .ok_or_else(|| {
                ILError::index(format!(
                    "Index not found for search query: {:?}",
                    self.query
                ))
            })?;

        let supported_fields = index_kind
            .dynamic_fields(index_def.as_ref())?
            .into_iter()
            .map(|field| (field.name().clone(), field))
            .collect::<HashMap<_, _>>();

        let mut fields = projected_schema
            .fields()
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        for name in &self.dynamic_fields {
            let field = supported_fields.get(name).cloned().ok_or_else(|| {
                ILError::invalid_input(format!(
                    "Unsupported dynamic field `{name}` for index `{}`",
                    index_def.name
                ))
            })?;
            fields.push(field);
        }

        Ok(Arc::new(Schema::new_with_metadata(
            fields,
            projected_schema.metadata().clone(),
        )))
    }
}

pub(crate) async fn process_search(
    table: &Table,
    search: TableSearch,
) -> ILResult<RecordBatchStream> {
    let index_kind = search.query.index_kind();

    let (index_def, index_kind) = table
        .index_manager
        .iter_index_and_kind()
        .find(|(index_def, _)| index_def.kind == index_kind)
        .ok_or_else(|| {
            ILError::index(format!(
                "Index not found for search query: {:?}",
                search.query
            ))
        })?;

    let catalog_helper = CatalogHelper::new(table.catalog.clone());

    let inline_index_records = catalog_helper
        .get_inline_indexes(&[index_def.index_id])
        .await?;

    let index_kind_captured = index_kind.clone();
    let index_def_captured = index_def.clone();
    let search_captured = search.clone();
    let inline_handle = tokio::spawn(async move {
        if inline_index_records.is_empty() {
            return Ok::<_, ILError>(SearchIndexEntries {
                row_ids: vec![],
                scores: vec![],
                score_higher_is_better: false,
                dynamic_columns: vec![],
            });
        }
        let inline_search_entries = search_inline_rows(
            index_kind_captured.as_ref(),
            &index_def_captured,
            &search_captured,
            inline_index_records,
        )
        .await?;
        Ok::<_, ILError>(inline_search_entries)
    });

    let data_file_records = catalog_helper.get_data_files(&table.table_id).await?;

    let index_id = index_def.index_id;
    let index_file_records = catalog_helper
        .get_index_files_by_index_id(&index_id)
        .await?;
    let index_file_records: HashMap<Uuid, IndexFileRecord> = index_file_records
        .into_iter()
        .map(|record| (record.data_file_id, record))
        .collect();

    let mut handles = Vec::new();
    for data_file_record in data_file_records.iter() {
        let data_file_id = data_file_record.data_file_id;
        let index_file_record =
            index_file_records
                .get(&data_file_id)
                .cloned()
                .ok_or(ILError::index(format!(
                    "Index file not found for index {index_id} and data file {data_file_id}"
                )))?;
        let storage = table.storage.clone();
        let index_kind = index_kind.clone();
        let index_def = index_def.clone();
        let search_query = search.query.clone();
        let dynamic_fields = search.dynamic_fields.clone();
        let handle = tokio::spawn(async move {
            let search_entries = search_index_file(
                storage.as_ref(),
                index_kind.as_ref(),
                &index_def,
                search_query.as_ref(),
                &index_file_record,
                &dynamic_fields,
            )
            .await?;
            Ok::<_, ILError>((data_file_id, search_entries))
        });
        handles.push(handle);
    }

    let join_all = futures::future::join_all(handles).await;

    let mut all_search_entries = Vec::with_capacity(join_all.len() + 1);
    for res in join_all {
        let (data_file_id, search_entries) = res??;
        all_search_entries.push((RowLocation::DataFile(data_file_id), search_entries));
    }

    let inline_search_entries = inline_handle.await??;
    let score_higher_is_better = inline_search_entries.score_higher_is_better;
    all_search_entries.push((RowLocation::Inline, inline_search_entries));

    let merged_entries = merge_search_index_entries(
        all_search_entries,
        score_higher_is_better,
        search.query.limit(),
    )?;

    let row_id_idx = table
        .table_schema
        .arrow_schema
        .index_of(INTERNAL_ROW_ID_FIELD_NAME)?;

    // Determine if user explicitly requested _row_id in their projection
    let user_wants_row_id = search
        .projection
        .as_ref()
        .map(|p| p.contains(&row_id_idx))
        .unwrap_or(true); // None means all columns, which includes _row_id

    // Build internal projection that always includes _row_id for internal processing
    // (sort_batches and gather_dynamic_columns_from_batch need _row_id at position 0)
    let internal_projection = if user_wants_row_id {
        search.projection.clone()
    } else {
        search.projection.as_ref().map(|p| {
            let mut proj = vec![row_id_idx];
            proj.extend(p.iter());
            proj
        })
    };

    let (inline_batch, data_file_batches) = read_rows(
        &catalog_helper,
        table,
        &merged_entries.row_score_locations,
        internal_projection,
        &data_file_records,
    )
    .await?;

    let sorted_batch = sort_batches(
        inline_batch,
        data_file_batches,
        &merged_entries.row_score_locations,
        score_higher_is_better,
    )?;
    let dynamic_columns =
        gather_dynamic_columns_from_batch(&sorted_batch, &merged_entries.dynamic_lookups)?;
    let final_batch = append_dynamic_columns_to_record_batch(&sorted_batch, &dynamic_columns)?;

    // Remove internal _row_id column from final output only if user didn't request it
    let final_batch = if user_wants_row_id {
        final_batch
    } else {
        remove_row_id_column(&final_batch)?
    };

    Ok(Box::pin(futures::stream::iter(vec![Ok(final_batch)])))
}

async fn search_inline_rows(
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search: &TableSearch,
    inline_index_records: Vec<InlineIndexRecord>,
) -> ILResult<SearchIndexEntries> {
    let mut index_builder = index_kind.builder(index_def)?;

    for record in inline_index_records {
        index_builder.read_bytes(&record.index_data)?;
    }

    let index = index_builder.build()?;

    let search_index_entries = index
        .search(search.query.as_ref(), &search.dynamic_fields)
        .await?;

    Ok(search_index_entries)
}

async fn search_index_file(
    storage: &dyn Storage,
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search_query: &dyn SearchQuery,
    index_file_record: &IndexFileRecord,
    dynamic_fields: &[String],
) -> ILResult<SearchIndexEntries> {
    let index_file = storage.open(&index_file_record.relative_path).await?;

    let mut index_builder = index_kind.builder(index_def)?;
    index_builder.read_file(index_file).await?;

    let index = index_builder.build()?;

    let search_index_entries = index.search(search_query, dynamic_fields).await?;

    Ok(search_index_entries)
}

fn merge_search_index_entries(
    search_entries: Vec<(RowLocation, SearchIndexEntries)>,
    score_higher_is_better: bool,
    limit: Option<usize>,
) -> ILResult<MergedSearchEntries> {
    let mut row_score_locations = Vec::new();
    let mut all_row_ids = Vec::new();
    let mut dynamic_column_groups = Vec::new();

    for (location, search_entries) in search_entries {
        all_row_ids.extend(search_entries.row_ids.iter().copied());
        dynamic_column_groups.push(search_entries.dynamic_columns.clone());
        for (row_id, score) in search_entries
            .row_ids
            .into_iter()
            .zip(search_entries.scores)
        {
            row_score_locations.push((RowScore { row_id, score }, location.clone()));
        }
    }

    if score_higher_is_better {
        row_score_locations.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        row_score_locations.sort_by(|a, b| {
            a.0.score
                .partial_cmp(&b.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    if let Some(limit) = limit {
        row_score_locations.truncate(limit);
    }

    let dynamic_columns = concat_dynamic_columns(&dynamic_column_groups)?;
    let dynamic_lookups = build_dynamic_column_lookups(&all_row_ids, &dynamic_columns)?;

    Ok(MergedSearchEntries {
        row_score_locations,
        dynamic_lookups,
    })
}

#[derive(Debug, Clone)]
struct RowScore {
    row_id: Uuid,
    score: f64,
}

#[derive(Debug, Clone)]
struct DynamicColumnLookup {
    field: FieldRef,
    row_id_to_index: HashMap<Uuid, u32>,
    values: ArrayRef,
}

#[derive(Debug, Clone)]
struct MergedSearchEntries {
    row_score_locations: Vec<(RowScore, RowLocation)>,
    dynamic_lookups: Vec<DynamicColumnLookup>,
}

#[derive(Debug, Clone)]
enum RowLocation {
    Inline,
    DataFile(Uuid),
}

fn concat_dynamic_columns(column_groups: &[Vec<DynamicColumn>]) -> ILResult<Vec<DynamicColumn>> {
    if column_groups.is_empty() {
        return Err(ILError::internal(
            "No search index entries found when concatenating dynamic columns",
        ));
    }

    let mut output = Vec::new();
    for (idx, expected_column) in column_groups[0].iter().enumerate() {
        let arrays = column_groups
            .iter()
            .map(|group| {
                let column = group.get(idx).ok_or_else(|| {
                    ILError::internal(format!(
                        "Dynamic column group missing column {}",
                        expected_column.field.name()
                    ))
                })?;
                if column.field.as_ref() != expected_column.field.as_ref() {
                    return Err(ILError::internal(format!(
                        "Dynamic column field mismatch: expected {:?}, got {:?}",
                        expected_column.field, column.field
                    )));
                }
                Ok(column.values.as_ref())
            })
            .collect::<ILResult<Vec<_>>>()?;
        let values = arrow::compute::concat(arrays.as_slice())?;
        output.push(DynamicColumn {
            field: expected_column.field.clone(),
            values,
        });
    }
    Ok(output)
}

fn build_dynamic_column_lookups(
    row_ids: &[Uuid],
    dynamic_columns: &[DynamicColumn],
) -> ILResult<Vec<DynamicColumnLookup>> {
    let mut row_id_to_index = HashMap::with_capacity(row_ids.len());
    for (idx, row_id) in row_ids.iter().enumerate() {
        if row_id_to_index.insert(*row_id, idx as u32).is_some() {
            return Err(ILError::internal(format!(
                "Duplicate row id {row_id} found while building dynamic column lookups"
            )));
        }
    }

    dynamic_columns
        .iter()
        .map(|column| {
            if column.values.len() != row_ids.len() {
                return Err(ILError::internal(format!(
                    "Dynamic column {} length {} does not match row count {}",
                    column.field.name(),
                    column.values.len(),
                    row_ids.len()
                )));
            }
            Ok(DynamicColumnLookup {
                field: column.field.clone(),
                row_id_to_index: row_id_to_index.clone(),
                values: column.values.clone(),
            })
        })
        .collect()
}

fn gather_dynamic_columns_from_batch(
    batch: &RecordBatch,
    lookups: &[DynamicColumnLookup],
) -> ILResult<Vec<DynamicColumn>> {
    if lookups.is_empty() {
        return Ok(Vec::new());
    }

    let row_ids = extract_row_ids_from_record_batch(batch)?;
    lookups
        .iter()
        .map(|lookup| {
            let indices = row_ids
                .iter()
                .map(|row_id| {
                    lookup.row_id_to_index.get(row_id).copied().ok_or_else(|| {
                        ILError::internal(format!(
                            "Dynamic column {} lookup missing row id {row_id}",
                            lookup.field.name()
                        ))
                    })
                })
                .collect::<ILResult<Vec<_>>>()?;
            let indices = arrow::array::UInt32Array::from(indices);
            let values = arrow::compute::take(lookup.values.as_ref(), &indices, None)?;
            Ok(DynamicColumn {
                field: lookup.field.clone(),
                values,
            })
        })
        .collect()
}

fn append_dynamic_columns_to_record_batch(
    batch: &RecordBatch,
    dynamic_columns: &[DynamicColumn],
) -> ILResult<RecordBatch> {
    if dynamic_columns.is_empty() {
        return Ok(batch.clone());
    }

    let mut fields = batch.schema().fields().iter().cloned().collect::<Vec<_>>();
    let mut arrays = batch.columns().to_vec();
    for column in dynamic_columns {
        if column.values.len() != batch.num_rows() {
            return Err(ILError::internal(format!(
                "Dynamic column {} length {} does not match batch row count {}",
                column.field.name(),
                column.values.len(),
                batch.num_rows()
            )));
        }
        fields.push(column.field.clone());
        arrays.push(column.values.clone());
    }
    let schema = Arc::new(Schema::new_with_metadata(
        fields,
        batch.schema().metadata().clone(),
    ));
    Ok(RecordBatch::try_new(schema, arrays)?)
}

fn remove_row_id_column(batch: &RecordBatch) -> ILResult<RecordBatch> {
    let schema = batch.schema();
    let row_id_idx = schema.index_of(INTERNAL_ROW_ID_FIELD_NAME)?;
    let mut fields = Vec::with_capacity(batch.num_columns() - 1);
    let mut columns = Vec::with_capacity(batch.num_columns() - 1);
    for (i, field) in schema.fields().iter().enumerate() {
        if i != row_id_idx {
            fields.push(field.clone());
            columns.push(batch.column(i).clone());
        }
    }
    let new_schema = Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
    RecordBatch::try_new(new_schema, columns)
        .map_err(|e| ILError::internal(format!("Failed to remove row id column: {}", e)))
}

/// Reorder batch columns to match the order of the target schema
fn reorder_batch_to_schema(batch: &RecordBatch, target_schema: &Schema) -> ILResult<RecordBatch> {
    let batch_schema = batch.schema();
    let mut columns = Vec::with_capacity(target_schema.fields.len());
    for target_field in target_schema.fields.iter() {
        let idx = batch_schema.index_of(target_field.name())?;
        columns.push(batch.column(idx).clone());
    }
    RecordBatch::try_new(Arc::new(target_schema.clone()), columns)
        .map_err(|e| ILError::internal(format!("Failed to reorder batch: {}", e)))
}

async fn read_rows(
    catalog_helper: &CatalogHelper,
    table: &Table,
    row_id_score_locations: &[(RowScore, RowLocation)],
    projection: Option<Vec<usize>>,
    data_file_records: &[DataFileRecord],
) -> ILResult<(RecordBatch, Vec<RecordBatch>)> {
    // Collect inline row ids
    let inline_row_ids: Vec<_> = row_id_score_locations
        .iter()
        .filter(|(_, location)| matches!(location, RowLocation::Inline))
        .map(|(row, _)| row.row_id)
        .collect();

    // Collect data file row ids grouped by data_file_id
    let mut data_file_row_ids: HashMap<Uuid, Vec<Uuid>> = HashMap::new();
    for (row, location) in row_id_score_locations {
        if let RowLocation::DataFile(data_file_id) = location {
            data_file_row_ids
                .entry(*data_file_id)
                .or_default()
                .push(row.row_id);
        }
    }

    // Use the provided projection directly (caller ensures _row_id is included if needed)
    let projected_schema = Arc::new(project_schema(
        &table.table_schema.arrow_schema,
        projection.as_ref(),
    )?);
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);

    // Create inline rows reading task
    let inline_task = async {
        if inline_row_ids.is_empty() {
            return rows_to_record_batch(&projected_schema, &[]);
        }
        let row_stream = catalog_helper
            .scan_inline_rows(
                &table.table_id,
                &catalog_schema,
                Some(&inline_row_ids),
                &[],
                None,
            )
            .await?;
        let rows: Vec<Row> = row_stream.try_collect::<Vec<_>>().await?;
        let batch = rows_to_record_batch(&projected_schema, &rows)?;
        Ok::<_, ILError>(batch)
    };

    // Create data file reading tasks - parallelized
    let data_file_tasks: Vec<_> = data_file_row_ids
        .into_iter()
        .map(|(data_file_id, row_ids)| {
            let table = table.clone();
            let projection = projection.clone();
            let data_file_records = data_file_records.to_vec();

            tokio::spawn(async move {
                let data_file_record = data_file_records
                    .iter()
                    .find(|record| record.data_file_id == data_file_id)
                    .ok_or_else(|| {
                        ILError::index(format!(
                            "Data file record not found for data file id {data_file_id}"
                        ))
                    })?;

                let stream = read_data_file_by_record(
                    table.storage.as_ref(),
                    &table.table_schema,
                    data_file_record,
                    projection,
                    vec![row_ids_in_list_expr(row_ids)],
                    1024,
                )
                .await?;

                let batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;
                Ok::<_, ILError>(batches)
            })
        })
        .collect();

    // Execute inline task and all data file tasks concurrently
    let (inline_result, data_file_results) =
        tokio::join!(inline_task, futures::future::join_all(data_file_tasks));

    let inline_batch = inline_result?;

    // Collect all data file batches
    let mut all_data_file_batches = Vec::new();
    for result in data_file_results {
        let batches = result.map_err(|e| ILError::internal(format!("Task join error: {e}")))??;
        all_data_file_batches.extend(batches);
    }

    Ok((inline_batch, all_data_file_batches))
}

fn sort_batches(
    inline_batch: RecordBatch,
    data_file_batches: Vec<RecordBatch>,
    row_id_score_locations: &[(RowScore, RowLocation)],
    score_higher_is_better: bool,
) -> ILResult<RecordBatch> {
    let mut batch = inline_batch;
    for data_file_batch in data_file_batches {
        // Ensure data_file_batch has the same column order as batch before concatenating
        let reordered = reorder_batch_to_schema(&data_file_batch, batch.schema().as_ref())?;
        batch = arrow::compute::concat_batches(&batch.schema(), [&batch, &reordered])?;
    }

    let mut scores = Vec::new();
    let row_id_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or(ILError::index("Row id column not found in batch"))?;
    for row_id in row_id_array.iter() {
        let row_id_bytes = row_id.ok_or(ILError::index("Row id is null"))?;
        let row_id = Uuid::from_slice(row_id_bytes)?;
        let row = row_id_score_locations
            .iter()
            .find(|(row, _)| row.row_id == row_id)
            .ok_or(ILError::index(format!(
                "Row id score not found for row id {row_id}"
            )))?;
        scores.push(row.0.score);
    }

    let scores_array = Float64Array::from(scores);

    let sort_options = SortOptions::default().with_descending(score_higher_is_better);
    let indices = arrow::compute::sort_to_indices(&scores_array, Some(sort_options), None)?;

    let batch = arrow::compute::take_record_batch(&batch, &indices)?;

    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{StringArray, UInt64Array};
    use arrow::datatypes::{DataType, Field};

    fn create_test_batch_with_row_id() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new(INTERNAL_ROW_ID_FIELD_NAME, DataType::UInt64, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("score", DataType::Float64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    fn create_test_batch_without_row_id() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("title", DataType::Utf8, false),
            Field::new("score", DataType::Float64, false),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
                Arc::new(arrow::array::Float64Array::from(vec![1.0, 2.0, 3.0])),
            ],
        )
        .unwrap()
    }

    #[test]
    fn test_remove_row_id_column() {
        let batch = create_test_batch_with_row_id();
        let result = remove_row_id_column(&batch).unwrap();

        assert_eq!(result.num_columns(), 2);
        assert_eq!(
            result
                .schema()
                .fields()
                .iter()
                .map(|f| f.name())
                .collect::<Vec<_>>(),
            vec!["title", "score"]
        );
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_remove_row_id_column_preserves_data() {
        let batch = create_test_batch_with_row_id();
        let result = remove_row_id_column(&batch).unwrap();

        let title_col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(title_col.value(0), "a");
        assert_eq!(title_col.value(1), "b");
        assert_eq!(title_col.value(2), "c");

        let score_col = result
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert_eq!(score_col.value(0), 1.0);
        assert_eq!(score_col.value(1), 2.0);
        assert_eq!(score_col.value(2), 3.0);
    }

    #[test]
    fn test_reorder_batch_to_schema() {
        let batch = create_test_batch_with_row_id();
        let target_schema = Schema::new(vec![
            Field::new("score", DataType::Float64, false),
            Field::new("title", DataType::Utf8, false),
            Field::new(INTERNAL_ROW_ID_FIELD_NAME, DataType::UInt64, false),
        ]);

        let result = reorder_batch_to_schema(&batch, &target_schema).unwrap();

        assert_eq!(result.num_columns(), 3);
        // Verify order matches target schema
        assert_eq!(result.schema().field(0).name(), "score");
        assert_eq!(result.schema().field(1).name(), "title");
        assert_eq!(result.schema().field(2).name(), INTERNAL_ROW_ID_FIELD_NAME);
    }

    #[test]
    fn test_reorder_batch_preserves_data() {
        let batch = create_test_batch_with_row_id();
        let target_schema = Schema::new(vec![
            Field::new("score", DataType::Float64, false),
            Field::new("title", DataType::Utf8, false),
            Field::new(INTERNAL_ROW_ID_FIELD_NAME, DataType::UInt64, false),
        ]);

        let result = reorder_batch_to_schema(&batch, &target_schema).unwrap();

        // Check score column (was at index 2, now at index 0)
        let score_col = result
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .unwrap();
        assert_eq!(score_col.value(0), 1.0);
    }

    #[test]
    fn test_user_wants_row_id_logic() {
        // Test case: projection excludes _row_id (index 0 in full schema with row_id)
        let projection = Some(vec![1]); // Only "title" column
        let row_id_idx = 0;

        let user_wants_row_id = projection
            .as_ref()
            .map(|p| p.contains(&row_id_idx))
            .unwrap_or(true);
        assert!(!user_wants_row_id);

        // Test case: projection includes _row_id
        let projection_with_row_id = Some(vec![0, 1]); // _row_id and title
        let user_wants_row_id = projection_with_row_id
            .as_ref()
            .map(|p| p.contains(&row_id_idx))
            .unwrap_or(true);
        assert!(user_wants_row_id);

        // Test case: None projection means all columns
        let projection_none: Option<Vec<usize>> = None;
        let user_wants_row_id = projection_none
            .as_ref()
            .map(|p| p.contains(&row_id_idx))
            .unwrap_or(true);
        assert!(user_wants_row_id);
    }

    #[test]
    fn test_internal_projection_includes_row_id_when_user_excludes() {
        let row_id_idx = 0;
        let user_projection = Some(vec![1]); // Only "title" (index 1 in full schema)

        // Simulating the logic in process_search
        let user_wants_row_id = user_projection
            .as_ref()
            .map(|p| p.contains(&row_id_idx))
            .unwrap_or(true);

        let internal_projection = if user_wants_row_id {
            user_projection.clone()
        } else {
            user_projection.as_ref().map(|p| {
                let mut proj = vec![row_id_idx];
                proj.extend(p.iter());
                proj
            })
        };

        // Internal projection should include row_id at position 0
        let internal_proj = internal_projection.unwrap();
        assert_eq!(internal_proj[0], row_id_idx);
        assert_eq!(internal_proj[1], 1); // title is now at position 1
        assert_eq!(internal_proj.len(), 2);
    }

    #[test]
    fn test_internal_projection_preserves_user_row_id_request() {
        let row_id_idx = 0;
        let user_projection = Some(vec![0, 1]); // Both _row_id and title

        let user_wants_row_id = user_projection
            .as_ref()
            .map(|p| p.contains(&row_id_idx))
            .unwrap_or(true);

        let internal_projection = if user_wants_row_id {
            user_projection.clone()
        } else {
            user_projection.as_ref().map(|p| {
                let mut proj = vec![row_id_idx];
                proj.extend(p.iter());
                proj
            })
        };

        // Should be unchanged since user already included row_id
        let internal_proj = internal_projection.unwrap();
        assert_eq!(internal_proj, vec![0, 1]);
    }

    #[test]
    fn test_batch_without_row_id_to_schema() {
        // Test reordering a batch that doesn't have row_id
        let batch = create_test_batch_without_row_id();
        let target_schema = Schema::new(vec![
            Field::new("score", DataType::Float64, false),
            Field::new("title", DataType::Utf8, false),
        ]);

        let result = reorder_batch_to_schema(&batch, &target_schema).unwrap();
        assert_eq!(result.num_columns(), 2);
        assert_eq!(result.schema().field(0).name(), "score");
        assert_eq!(result.schema().field(1).name(), "title");
    }
}
