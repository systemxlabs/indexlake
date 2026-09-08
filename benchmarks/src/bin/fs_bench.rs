// FS benchmark for indexlake read-path comparison (untracked helper, not part of the repo).
// Compares scan performance between read-path implementations on the same machine.
// Uses a postgres catalog (localhost:5432 postgres/password, dbname=postgres) + filesystem storage.
//
// Env knobs: IL_BENCH_ROWS (default 100000), IL_BENCH_TASKS (10), IL_BENCH_ROUNDS (5),
//            IL_BENCH_INLINE_LIMIT (1000), IL_BENCH_ROW_GROUP_SIZE (8192),
//            IL_BENCH_DIST (uniform|range)
//
// Output lines: "RESULT phase=<name> run=<n> rows=<r> ms=<m> rows_per_s=<x>" for machine parsing.
use std::time::{Duration, Instant};

use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use futures::StreamExt;
use indexlake::expr::{col, lit};
use indexlake::table::{TableConfig, TableCreation, TableInsertion, TableScan, TableScanPartition};
use indexlake::{Client, ILError};
use indexlake_integration_tests::storage_fs;
use std::sync::Arc;

async fn pick_catalog() -> Arc<dyn indexlake::catalog::Catalog> {
    // Reuse the already-running postgres; avoid the docker-compose up in
    // indexlake_integration_tests::catalog_postgres.
    let builder = indexlake_catalog_postgres::PostgresCatalogBuilder::new(
        "localhost",
        5432,
        "postgres",
        "password",
    )
    .dbname("postgres")
    .pool_max_size(100)
    .pool_idle_timeout(Some(std::time::Duration::from_secs(10)));
    Arc::new(builder.build().await.unwrap())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn bench_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("grp", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]))
}

/// uniform: grp = id % 100 (every file spans the full value domain, nothing is
/// prunable). range: grp = (id / 200) % 100 (each ~1000-row file covers only 5
/// consecutive grp values, so per-file min/max pruning can skip most files).
fn new_record_batch(start_id: i64, num_rows: usize, dist: &str) -> RecordBatch {
    let ids: Vec<i64> = (start_id..start_id + num_rows as i64).collect();
    let grps: Vec<i64> = ids
        .iter()
        .map(|v| match dist {
            "range" => (v / 200) % 100,
            _ => v % 100,
        })
        .collect();
    let names: Vec<String> = ids.iter().map(|v| format!("name-{v:020}")).collect();
    let values: Vec<f64> = ids.iter().map(|v| *v as f64 * 1.5).collect();
    RecordBatch::try_new(
        bench_schema(),
        vec![
            Arc::new(Int64Array::from(ids)) as _,
            Arc::new(Int64Array::from(grps)) as _,
            Arc::new(StringArray::from(names)) as _,
            Arc::new(Float64Array::from(values)) as _,
        ],
    )
    .unwrap()
}

async fn run_scan_timed(table: &indexlake::table::Table, scan: TableScan) -> (usize, Duration) {
    let start = Instant::now();
    let stream = table.scan(scan).await.unwrap();
    let mut rows = 0usize;
    let mut batches = stream;
    while let Some(batch) = batches.next().await {
        rows += batch.unwrap().num_rows();
    }
    (rows, start.elapsed())
}

fn report(phase: &str, run: usize, rows: usize, elapsed: Duration) {
    let ms = elapsed.as_secs_f64() * 1000.0;
    let rps = rows as f64 / elapsed.as_secs_f64();
    println!("RESULT phase={phase} run={run} rows={rows} ms={ms:.1} rows_per_s={rps:.0}");
}

async fn wait_data_files_settled(
    table: &indexlake::table::Table,
    expected_min: usize,
    timeout: Duration,
) {
    let poll = Duration::from_secs(2);
    let mut last_count = table.data_file_count().await.unwrap();
    let mut stable_for = Duration::ZERO;
    let start = Instant::now();
    loop {
        tokio::time::sleep(poll).await;
        let count = table.data_file_count().await.unwrap();
        if count != last_count {
            last_count = count;
            stable_for = Duration::ZERO;
        } else {
            stable_for += poll;
        }
        if last_count >= expected_min && stable_for >= Duration::from_secs(4) {
            break;
        }
        if start.elapsed() > timeout {
            println!(
                "benchmark: WARNING data file count not settled (last={last_count}, expected_min={expected_min}), continuing"
            );
            break;
        }
    }
    println!("benchmark: data file count = {last_count}");
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let total_rows = env_usize("IL_BENCH_ROWS", 100_000);
    let num_tasks = env_usize("IL_BENCH_TASKS", 10);
    let rounds = env_usize("IL_BENCH_ROUNDS", 5);
    let inline_limit = env_usize("IL_BENCH_INLINE_LIMIT", 1000);
    let row_group_size = env_usize("IL_BENCH_ROW_GROUP_SIZE", 8192);
    let dist = std::env::var("IL_BENCH_DIST").unwrap_or_else(|_| "uniform".into());
    let batch_size = total_rows / num_tasks / 10;
    println!(
        "benchmark: rows={total_rows} tasks={num_tasks} inline_limit={inline_limit} row_group={row_group_size} dist={dist}"
    );

    let catalog = pick_catalog().await;
    let storage = storage_fs();
    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: bench_schema(),
        config: TableConfig {
            inline_row_count_limit: inline_limit,
            parquet_row_group_size: row_group_size,
            preferred_data_file_format: indexlake::storage::DataFileFormat::ParquetV2,
        },
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    // Concurrent insert: each task slices its row range into batches and
    // inserts them in parallel. The remainder of total_rows / num_tasks is
    // spread over the first tasks so no row is dropped.
    let start_time = Instant::now();
    let mut handles = Vec::new();
    let rows_per_task = total_rows / num_tasks;
    let remainder = total_rows % num_tasks;
    let mut start_row = 0usize;
    for task in 0..num_tasks {
        let table = table.clone();
        let task_rows = rows_per_task + usize::from(task < remainder);
        let base = start_row as i64;
        start_row += task_rows;
        let mut handles_inner = Vec::new();
        let mut offset = 0usize;
        while offset < task_rows {
            let n = batch_size.min(task_rows - offset);
            let batch = new_record_batch(base + offset as i64, n, &dist);
            let table = table.clone();
            handles_inner.push(tokio::spawn(async move {
                table.insert(TableInsertion::new(vec![batch])).await
            }));
            offset += n;
        }
        handles.push(async move {
            for h in handles_inner {
                h.await.unwrap()?;
            }
            Ok::<(), ILError>(())
        });
    }
    for h in handles {
        h.await?;
    }
    let insert_elapsed = start_time.elapsed();
    println!(
        "RESULT phase=insert run=0 rows={total_rows} ms={:.1} rows_per_s={:.0}",
        insert_elapsed.as_secs_f64() * 1000.0,
        total_rows as f64 / insert_elapsed.as_secs_f64()
    );

    // Wait for the async dump to settle inline rows into parquet data files.
    wait_data_files_settled(&table, total_rows / inline_limit, Duration::from_secs(180)).await;

    // Count sanity check before timing the scan phases.
    let counts = table
        .count(&[TableScanPartition::single_partition()])
        .await?;
    assert_eq!(counts[0], total_rows, "count mismatch");

    // Full-column scan.
    for run in 0..rounds {
        let scan = TableScan::default().with_partition(TableScanPartition::single_partition());
        let (rows, elapsed) = run_scan_timed(&table, scan).await;
        assert_eq!(rows, total_rows);
        report("scan_full", run, rows, elapsed);
    }

    // Projection scan (id, value). Projection indices are over the table
    // schema with the implicit row_id column at index 0, so 1 = id, 4 = value.
    for run in 0..rounds {
        let scan = TableScan::default()
            .with_partition(TableScanPartition::single_partition())
            .with_projection(Some(vec![1, 4]));
        let (rows, elapsed) = run_scan_timed(&table, scan).await;
        assert_eq!(rows, total_rows);
        report("scan_projection", run, rows, elapsed);
    }

    // Equality filter scan (grp == 42): 1% selectivity, exercises row-level
    // predicate filtering and row-group stats pruning. The assertion only
    // holds when the id domain covers all 100 grp buckets (range dist needs
    // total_rows >= 20000); with uniform dist every bucket has exactly
    // total_rows / 100 rows regardless.
    for run in 0..rounds {
        let scan = TableScan::default()
            .with_partition(TableScanPartition::single_partition())
            .with_filters(vec![col("grp").eq(lit(42i64))]);
        let (rows, elapsed) = run_scan_timed(&table, scan).await;
        report("scan_filter", run, rows, elapsed);
        assert_eq!(rows, total_rows / 100, "filter scan row mismatch: {rows}");
    }

    // Range filter scan (id >= floor(total/2)): ~50% selectivity, and the id
    // column min/max stats support file/row-group pruning. The expected count
    // is total - floor(total/2) (= ceil for odd totals) so the assertion holds
    // regardless of parity.
    let range_threshold = (total_rows / 2) as i64;
    let range_expected = total_rows - range_threshold as usize;
    for run in 0..rounds {
        let scan = TableScan::default()
            .with_partition(TableScanPartition::single_partition())
            .with_filters(vec![col("id").gteq(lit(range_threshold))]);
        let (rows, elapsed) = run_scan_timed(&table, scan).await;
        report("scan_filter_range", run, rows, elapsed);
        assert_eq!(
            rows, range_expected,
            "filter range scan row mismatch: {rows}"
        );
    }

    println!("benchmark: done");
    Ok(())
}
