use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{FixedSizeBinaryArray, Float64Array, RecordBatch};
use arrow::compute::SortOptions;
use futures::TryStreamExt;
use uuid::Uuid;

use crate::catalog::{
    CatalogHelper, CatalogSchema, DataFileRecord, IndexFileRecord, Row, rows_to_record_batch,
};
use crate::expr::row_ids_in_list_expr;
use crate::index::{
    IndexColumnRequest, IndexDefinitionRef, IndexKind, IndexResultColumn, IndexResultOptions,
    RequestedIndexColumn, RowIdScore, SearchIndexEntries, SearchQuery,
};
use crate::storage::{Storage, read_data_file_by_record};
use crate::table::Table;
use crate::utils::{
    append_columns_to_record_batch, concat_index_result_columns, project_schema,
    reorder_index_result_columns, validate_index_result_columns,
};
use crate::{ILError, ILResult, RecordBatchStream};
use arrow_schema::FieldRef;

#[derive(Debug, Clone)]
pub struct TableSearch {
    pub query: Arc<dyn SearchQuery>,
    pub projection: Option<Vec<usize>>,
    pub index_columns: Vec<IndexColumnRequest>,
}

pub(crate) async fn process_search(
    table: &Table,
    search: TableSearch,
) -> ILResult<RecordBatchStream> {
    let index_kind_name = search.query.index_kind();

    let (index_def, index_kind) = table
        .index_manager
        .iter_index_and_kind()
        .find(|(index_def, _)| index_def.kind == index_kind_name)
        .ok_or_else(|| {
            ILError::index(format!(
                "Index not found for search query: {:?}",
                search.query
            ))
        })?;

    let requested_columns =
        resolve_search_columns(table, &search, index_def.as_ref(), index_kind.as_ref())?;
    let dynamic_fields = index_kind.output_fields(index_def.as_ref(), &requested_columns)?;
    let result_options = IndexResultOptions {
        columns: requested_columns,
    };

    let catalog_helper = CatalogHelper::new(table.catalog.clone());

    let catalog_helper_captured = catalog_helper.clone();
    let index_kind_captured = index_kind.clone();
    let index_def_captured = index_def.clone();
    let search_captured = search.clone();
    let result_options_captured = result_options.clone();
    let dynamic_fields_captured = dynamic_fields.clone();
    let inline_handle = tokio::spawn(async move {
        let inline_search_entries = search_inline_rows(
            &catalog_helper_captured,
            index_kind_captured.as_ref(),
            &index_def_captured,
            &search_captured,
            &result_options_captured,
            &dynamic_fields_captured,
        )
        .await?;
        Ok::<_, ILError>(inline_search_entries)
    });

    let data_file_records = catalog_helper.get_data_files(&table.table_id).await?;

    let index_id = index_def.index_id;

    let mut handles = Vec::new();
    for data_file_record in data_file_records.iter() {
        let data_file_id = data_file_record.data_file_id;
        let storage = table.storage.clone();
        let catalog_helper = catalog_helper.clone();
        let index_kind = index_kind.clone();
        let index_def = index_def.clone();
        let search_query = search.query.clone();
        let result_options = result_options.clone();
        let dynamic_fields = dynamic_fields.clone();
        let handle = tokio::spawn(async move {
            let index_file_record = catalog_helper
                .get_index_file_by_index_id_and_data_file_id(&index_id, &data_file_id)
                .await?
                .ok_or(ILError::index(format!(
                    "Index file not found for index {index_id} and data file {data_file_id}"
                )))?;

            let search_entries = search_index_file(
                storage.as_ref(),
                index_kind.as_ref(),
                &index_def,
                search_query.as_ref(),
                &index_file_record,
                &result_options,
                &dynamic_fields,
            )
            .await?;
            Ok::<_, ILError>((data_file_id, search_entries))
        });
        handles.push(handle);
    }

    let join_all = futures::future::join_all(handles).await;

    let mut search_entries = Vec::with_capacity(join_all.len() + 1);
    let inline_search_entries = inline_handle.await??;
    search_entries.push((RowLocation::Inline, inline_search_entries));

    for res in join_all {
        let (data_file_id, entries) = res??;
        search_entries.push((RowLocation::DataFile(data_file_id), entries));
    }

    let merged_entries = merge_search_index_entries(search_entries, &dynamic_fields)?;

    let inline_batch = read_inline_rows(
        &catalog_helper,
        table,
        &merged_entries.row_id_score_locations,
        search.projection.clone(),
    )
    .await?;

    let data_file_batches = read_data_file_rows(
        table,
        &merged_entries.row_id_score_locations,
        search.projection.clone(),
        &data_file_records,
    )
    .await?;

    let (sorted_batch, sort_indices) = sort_batches(
        inline_batch,
        data_file_batches,
        &merged_entries.row_id_score_locations,
        merged_entries.score_higher_is_better,
        search.query.limit(),
    )?;
    let sorted_dynamic_columns =
        reorder_index_result_columns(&merged_entries.dynamic_columns, &sort_indices)?;
    let final_batch = append_columns_to_record_batch(&sorted_batch, &sorted_dynamic_columns)?;

    Ok(Box::pin(futures::stream::iter(vec![Ok(final_batch)])))
}

fn resolve_search_columns(
    table: &Table,
    search: &TableSearch,
    index_def: &crate::index::IndexDefinition,
    index_kind: &dyn IndexKind,
) -> ILResult<Vec<RequestedIndexColumn>> {
    let projected_schema = project_schema(&table.output_schema, search.projection.as_ref())?;
    let mut output_names = projected_schema
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect::<std::collections::HashSet<_>>();

    let requested_columns = search
        .index_columns
        .iter()
        .map(|column| {
            if let Some(index_name) = &column.index_name
                && index_name != &index_def.name
            {
                return Err(ILError::invalid_input(format!(
                    "Search result column `{}` targets index `{index_name}`, but the search uses index `{}`",
                    column.name, index_def.name
                )));
            }

            let output_name = column.output_name().to_string();
            if !output_names.insert(output_name.clone()) {
                return Err(ILError::invalid_input(format!(
                    "Duplicate output column name `{output_name}` in search result"
                )));
            }

            Ok(RequestedIndexColumn {
                name: column.name.clone(),
                output_name,
            })
        })
        .collect::<ILResult<Vec<_>>>()?;

    let _ = index_kind.output_fields(index_def, &requested_columns)?;
    Ok(requested_columns)
}

async fn search_inline_rows(
    catalog_helper: &CatalogHelper,
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search: &TableSearch,
    result_options: &IndexResultOptions,
    dynamic_fields: &[FieldRef],
) -> ILResult<SearchIndexEntries> {
    let mut index_builder = index_kind.builder(index_def)?;

    let inline_index_records = catalog_helper
        .get_inline_indexes(&[index_def.index_id])
        .await?;

    for record in inline_index_records {
        index_builder.read_bytes(&record.index_data)?;
    }

    let index = index_builder.build()?;

    let search_index_entries = index.search(search.query.as_ref(), result_options).await?;
    validate_index_result_columns(
        &search_index_entries.dynamic_columns,
        dynamic_fields,
        search_index_entries.row_id_scores.len(),
    )?;

    Ok(search_index_entries)
}

async fn search_index_file(
    storage: &dyn Storage,
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search_query: &dyn SearchQuery,
    index_file_record: &IndexFileRecord,
    result_options: &IndexResultOptions,
    dynamic_fields: &[FieldRef],
) -> ILResult<SearchIndexEntries> {
    let index_file = storage.open(&index_file_record.relative_path).await?;

    let mut index_builder = index_kind.builder(index_def)?;
    index_builder.read_file(index_file).await?;

    let index = index_builder.build()?;

    let search_index_entries = index.search(search_query, result_options).await?;
    validate_index_result_columns(
        &search_index_entries.dynamic_columns,
        dynamic_fields,
        search_index_entries.row_id_scores.len(),
    )?;

    Ok(search_index_entries)
}

fn merge_search_index_entries(
    search_entries: Vec<(RowLocation, SearchIndexEntries)>,
    dynamic_fields: &[FieldRef],
) -> ILResult<MergedSearchEntries> {
    let mut row_id_score_locations = Vec::new();
    let mut dynamic_column_groups = Vec::new();
    let mut score_higher_is_better = None;

    for (location, search_entries) in search_entries {
        if let Some(current) = score_higher_is_better {
            if current != search_entries.score_higher_is_better {
                return Err(ILError::internal(
                    "Search index entries disagree on score ordering",
                ));
            }
        } else {
            score_higher_is_better = Some(search_entries.score_higher_is_better);
        }

        validate_index_result_columns(
            &search_entries.dynamic_columns,
            dynamic_fields,
            search_entries.row_id_scores.len(),
        )?;

        dynamic_column_groups.push(search_entries.dynamic_columns.clone());
        for row_id_score in search_entries.row_id_scores {
            row_id_score_locations.push((row_id_score, location.clone()));
        }
    }

    let dynamic_columns = concat_index_result_columns(&dynamic_column_groups, dynamic_fields)?;

    Ok(MergedSearchEntries {
        row_id_score_locations,
        dynamic_columns,
        score_higher_is_better: score_higher_is_better.unwrap_or(true),
    })
}

#[derive(Debug, Clone)]
struct MergedSearchEntries {
    row_id_score_locations: Vec<(RowIdScore, RowLocation)>,
    dynamic_columns: Vec<IndexResultColumn>,
    score_higher_is_better: bool,
}

#[derive(Debug, Clone)]
enum RowLocation {
    Inline,
    DataFile(Uuid),
}

async fn read_inline_rows(
    catalog_helper: &CatalogHelper,
    table: &Table,
    row_id_score_locations: &[(RowIdScore, RowLocation)],
    projection: Option<Vec<usize>>,
) -> ILResult<RecordBatch> {
    let inline_row_ids = row_id_score_locations
        .iter()
        .filter(|(_, location)| matches!(location, RowLocation::Inline))
        .map(|(row_id_score, _)| row_id_score.row_id)
        .collect::<Vec<_>>();

    let projected_schema = Arc::new(project_schema(
        &table.table_schema.arrow_schema,
        projection.as_ref(),
    )?);
    let catalog_schema = Arc::new(CatalogSchema::from_arrow(&projected_schema)?);

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

    Ok(batch)
}

async fn read_data_file_rows(
    table: &Table,
    row_id_score_locations: &[(RowIdScore, RowLocation)],
    projection: Option<Vec<usize>>,
    data_file_records: &[DataFileRecord],
) -> ILResult<Vec<RecordBatch>> {
    let mut data_file_row_ids = HashMap::new();
    for (row_id_score, location) in row_id_score_locations {
        match location {
            RowLocation::Inline => {}
            RowLocation::DataFile(data_file_id) => {
                data_file_row_ids
                    .entry(*data_file_id)
                    .or_insert(Vec::new())
                    .push(row_id_score.row_id);
            }
        }
    }

    let mut all_batches = Vec::new();
    for (data_file_id, row_ids) in data_file_row_ids {
        let data_file_record = data_file_records
            .iter()
            .find(|record| record.data_file_id == data_file_id)
            .ok_or(ILError::index(format!(
                "Data file record not found for data file id {data_file_id}"
            )))?;
        let stream = read_data_file_by_record(
            table.storage.as_ref(),
            &table.table_schema,
            data_file_record,
            projection.clone(),
            vec![row_ids_in_list_expr(row_ids)],
            1024,
        )
        .await?;
        let batches: Vec<RecordBatch> = stream.try_collect::<Vec<_>>().await?;
        all_batches.extend(batches);
    }

    Ok(all_batches)
}

fn sort_batches(
    inline_batch: RecordBatch,
    data_file_batches: Vec<RecordBatch>,
    row_id_score_locations: &[(RowIdScore, RowLocation)],
    score_higher_is_better: bool,
    limit: Option<usize>,
) -> ILResult<(RecordBatch, arrow::array::UInt32Array)> {
    let mut batch = inline_batch;
    for data_file_batch in data_file_batches {
        batch = arrow::compute::concat_batches(&batch.schema(), [&batch, &data_file_batch])?;
    }

    let mut row_id_to_score = HashMap::with_capacity(row_id_score_locations.len());
    for (row_id_score, _) in row_id_score_locations {
        row_id_to_score.insert(row_id_score.row_id, row_id_score.score);
    }

    let mut scores = Vec::with_capacity(batch.num_rows());
    let row_id_array = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or(ILError::index("Row id column not found in batch"))?;
    for row_id in row_id_array.iter() {
        let row_id_bytes = row_id.ok_or(ILError::index("Row id is null"))?;
        let row_id = Uuid::from_slice(row_id_bytes)?;
        let score = row_id_to_score
            .get(&row_id)
            .copied()
            .ok_or(ILError::index(format!(
                "Row id score not found for row id {row_id}"
            )))?;
        scores.push(score);
    }

    let scores_array = Float64Array::from(scores);

    let sort_options = SortOptions::default().with_descending(score_higher_is_better);
    let mut indices = arrow::compute::sort_to_indices(&scores_array, Some(sort_options), None)?;
    if let Some(limit) = limit
        && indices.len() > limit
    {
        indices = arrow::array::UInt32Array::from(indices.values()[0..limit].to_vec());
    }

    let batch = arrow::compute::take_record_batch(&batch, &indices)?;

    Ok((batch, indices))
}
