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
use crate::index::{IndexDefinitionRef, IndexKind, RowIdScore, SearchIndexEntries, SearchQuery};
use crate::storage::{Storage, read_data_file_by_record};
use crate::table::Table;
use crate::utils::project_schema;
use crate::{ILError, ILResult, RecordBatchStream};

#[derive(Debug, Clone)]
pub struct TableSearch {
    pub query: Arc<dyn SearchQuery>,
    pub projection: Option<Vec<usize>>,
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

    let catalog_helper_captured = catalog_helper.clone();
    let index_kind_captured = index_kind.clone();
    let index_def_captured = index_def.clone();
    let search_captured = search.clone();
    let inline_handle = tokio::spawn(async move {
        let inline_search_entries = search_inline_rows(
            &catalog_helper_captured,
            index_kind_captured.as_ref(),
            &index_def_captured,
            &search_captured,
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
            )
            .await?;
            Ok::<_, ILError>((data_file_id, search_entries))
        });
        handles.push(handle);
    }

    let join_all = futures::future::join_all(handles).await;

    let mut index_file_search_entries = HashMap::with_capacity(join_all.len());
    for res in join_all {
        let (data_file_id, search_entries) = res??;
        index_file_search_entries.insert(data_file_id, search_entries);
    }

    let inline_search_entries = inline_handle.await??;

    let score_higher_is_better = inline_search_entries.score_higher_is_better;

    let row_id_score_locations = merge_search_index_entries(
        inline_search_entries,
        index_file_search_entries,
        search.query.limit(),
    )?;

    let inline_batch = read_inline_rows(
        &catalog_helper,
        table,
        &row_id_score_locations,
        search.projection.clone(),
    )
    .await?;

    let data_file_batches = read_data_file_rows(
        table,
        &row_id_score_locations,
        search.projection.clone(),
        &data_file_records,
    )
    .await?;

    let sorted_batch = sort_batches(
        inline_batch,
        data_file_batches,
        &row_id_score_locations,
        score_higher_is_better,
    )?;

    Ok(Box::pin(futures::stream::iter(vec![Ok(sorted_batch)])))
}

async fn search_inline_rows(
    catalog_helper: &CatalogHelper,
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search: &TableSearch,
) -> ILResult<SearchIndexEntries> {
    let mut index_builder = index_kind.builder(index_def)?;

    let inline_index_records = catalog_helper
        .get_inline_indexes(&[index_def.index_id])
        .await?;

    for record in inline_index_records {
        index_builder.read_bytes(&record.index_data)?;
    }

    let index = index_builder.build()?;

    let search_index_entries = index.search(search.query.as_ref()).await?;

    Ok(search_index_entries)
}

async fn search_index_file(
    storage: &dyn Storage,
    index_kind: &dyn IndexKind,
    index_def: &IndexDefinitionRef,
    search_query: &dyn SearchQuery,
    index_file_record: &IndexFileRecord,
) -> ILResult<SearchIndexEntries> {
    let index_file = storage.open(&index_file_record.relative_path).await?;

    let mut index_builder = index_kind.builder(index_def)?;
    index_builder.read_file(index_file).await?;

    let index = index_builder.build()?;

    let search_index_entries = index.search(search_query).await?;

    Ok(search_index_entries)
}

fn merge_search_index_entries(
    inline_search_entries: SearchIndexEntries,
    index_file_search_entries: HashMap<Uuid, SearchIndexEntries>,
    limit: Option<usize>,
) -> ILResult<Vec<(RowIdScore, RowLocation)>> {
    let mut row_id_score_locations = Vec::new();
    for row_id_score in inline_search_entries.row_id_scores.into_iter() {
        row_id_score_locations.push((row_id_score, RowLocation::Inline));
    }
    for (data_file_id, search_index_entries) in index_file_search_entries.into_iter() {
        for row_id_score in search_index_entries.row_id_scores.into_iter() {
            row_id_score_locations.push((row_id_score, RowLocation::DataFile(data_file_id)));
        }
    }

    if inline_search_entries.score_higher_is_better {
        row_id_score_locations.sort_by(|a, b| {
            b.0.score
                .partial_cmp(&a.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    } else {
        row_id_score_locations.sort_by(|a, b| {
            a.0.score
                .partial_cmp(&b.0.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    if let Some(limit) = limit {
        row_id_score_locations.truncate(limit);
    }

    Ok(row_id_score_locations)
}

#[derive(Debug)]
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
) -> ILResult<RecordBatch> {
    let mut batch = inline_batch;
    for data_file_batch in data_file_batches {
        batch = arrow::compute::concat_batches(&batch.schema(), [&batch, &data_file_batch])?;
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
        let row_id_score = row_id_score_locations
            .iter()
            .find(|(row_id_score, _)| row_id_score.row_id == row_id)
            .ok_or(ILError::index(format!(
                "Row id score not found for row id {row_id}"
            )))?;
        scores.push(row_id_score.0.score);
    }

    let scores_array = Float64Array::from(scores);

    let sort_options = SortOptions::default().with_descending(score_higher_is_better);
    let indices = arrow::compute::sort_to_indices(&scores_array, Some(sort_options), None)?;

    let batch = arrow::compute::take_record_batch(&batch, &indices)?;

    Ok(batch)
}
