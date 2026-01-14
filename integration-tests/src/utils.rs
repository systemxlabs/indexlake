use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::array::{ArrayRef, FixedSizeBinaryArray, RecordBatch, RecordBatchOptions};
use arrow::util::pretty::pretty_format_batches_with_schema;
use datafusion::physical_plan::collect;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::SessionContext;
use futures::TryStreamExt;
use indexlake::catalog::INTERNAL_ROW_ID_FIELD_NAME;
use indexlake::table::{Table, TableScan, TableSearch};
use indexlake::utils::{project_schema, schema_without_row_id};
use indexlake::{ILError, ILResult};

pub fn sort_record_batches(batches: &[RecordBatch], sort_col: &str) -> ILResult<RecordBatch> {
    if batches.is_empty() {
        return Err(ILError::invalid_input("record batches is empty"));
    }
    let record = arrow::compute::concat_batches(&batches[0].schema(), batches)?;

    let sort_col_idx = record.schema().index_of(sort_col)?;
    let sort_array = record.column(sort_col_idx);

    let indices = arrow::compute::sort_to_indices(sort_array, None, None)?;

    let sorted_arrays: Vec<ArrayRef> = record
        .columns()
        .iter()
        .map(|col| arrow::compute::take(col, &indices, None))
        .collect::<arrow::error::Result<_>>()?;

    let options = RecordBatchOptions::new().with_row_count(Some(record.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        record.schema(),
        sorted_arrays,
        &options,
    )?)
}

pub async fn full_table_scan(table: &Table) -> ILResult<String> {
    table_scan(table, TableScan::default()).await
}

pub async fn table_scan(table: &Table, scan: TableScan) -> ILResult<String> {
    let batch_schema = scan.output_schema(&table.output_schema)?;
    let batch_schema = Arc::new(schema_without_row_id(&batch_schema));

    let stream = table.scan(scan).await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    let mut sorted_batches = if batches.is_empty() {
        vec![]
    } else {
        vec![sort_record_batches(&batches, INTERNAL_ROW_ID_FIELD_NAME)?]
    };

    for batch in sorted_batches.iter_mut() {
        let Ok(idx) = batch.schema().index_of(INTERNAL_ROW_ID_FIELD_NAME) else {
            continue;
        };
        batch.remove_column(idx);
    }

    let table_str = pretty_format_batches_with_schema(batch_schema, &sorted_batches)?.to_string();
    Ok(table_str)
}

pub async fn table_search(table: &Table, search: TableSearch) -> ILResult<String> {
    let batch_schema = Arc::new(project_schema(
        &table.output_schema,
        search.projection.as_ref(),
    )?);
    let batch_schema = Arc::new(schema_without_row_id(&batch_schema));

    let stream = table.search(search).await?;
    let mut batches = stream.try_collect::<Vec<_>>().await?;

    for batch in batches.iter_mut() {
        let Ok(idx) = batch.schema().index_of(INTERNAL_ROW_ID_FIELD_NAME) else {
            continue;
        };
        batch.remove_column(idx);
    }

    let table_str = pretty_format_batches_with_schema(batch_schema, &batches)?.to_string();
    Ok(table_str)
}

pub async fn datafusion_insert(ctx: &SessionContext, sql: &str) -> String {
    datafusion_exec_and_sort(ctx, sql, None).await
}

pub async fn datafusion_scan(ctx: &SessionContext, sql: &str) -> String {
    datafusion_exec_and_sort(ctx, sql, Some(INTERNAL_ROW_ID_FIELD_NAME)).await
}

pub async fn datafusion_exec_and_sort(
    ctx: &SessionContext,
    sql: &str,
    sort_col: Option<&str>,
) -> String {
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    println!(
        "plan: {}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );

    let plan_schema = plan.schema();
    let plan_schema = Arc::new(schema_without_row_id(&plan_schema));

    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    let mut sorted_batches = if batches.is_empty() {
        vec![]
    } else if let Some(sort_col) = sort_col {
        vec![sort_record_batches(&batches, sort_col).unwrap()]
    } else {
        batches
    };

    for batch in sorted_batches.iter_mut() {
        let Ok(idx) = batch.schema().index_of(INTERNAL_ROW_ID_FIELD_NAME) else {
            continue;
        };
        batch.remove_column(idx);
    }

    let result_str = pretty_format_batches_with_schema(plan_schema, &sorted_batches)
        .unwrap()
        .to_string();
    println!("{}", result_str);
    result_str
}

pub async fn read_first_row_id_bytes_from_table(table: &Table) -> ILResult<Vec<u8>> {
    let stream = table.scan(TableScan::default()).await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    if batches.is_empty() {
        return Err(ILError::invalid_input("batches is empty"));
    }
    let sorted_batch = sort_record_batches(&batches, INTERNAL_ROW_ID_FIELD_NAME)?;
    let first_row_id_bytes = sorted_batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .unwrap()
        .value(0);
    if first_row_id_bytes.len() != 16 {
        return Err(ILError::invalid_input("first row id bytes is not 16 bytes"));
    }
    Ok(first_row_id_bytes.to_vec())
}

pub async fn assert_inline_row_count(table: &Table, check: impl Fn(usize) -> bool) -> ILResult<()> {
    let inline_row_count = table.inline_row_count().await?;
    if !check(inline_row_count) {
        return Err(ILError::internal("table inline row count check failed"));
    }
    Ok(())
}

pub async fn assert_data_file_count(table: &Table, check: impl Fn(usize) -> bool) -> ILResult<()> {
    let data_file_count = table.data_file_count().await?;
    if !check(data_file_count) {
        return Err(ILError::internal("table data file count check failed"));
    }
    Ok(())
}

pub fn timestamp_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time went backwards")
        .as_millis() as i64
}
