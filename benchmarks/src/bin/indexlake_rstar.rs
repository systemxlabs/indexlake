use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::AsArray;
use arrow::datatypes::DataType;
use futures::StreamExt;
use indexlake::Client;
use indexlake::ILError;
use indexlake::expr::{col, func, lit};
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableScan};
use indexlake_benchmarks::data::{arrow_rstar_table_schema, new_rstar_record_batch};
use indexlake_index_rstar::{RStarIndexKind, RStarIndexParams, WkbDialect};
use indexlake_integration_tests::{catalog_postgres, init_env_logger, storage_s3};

const QUERY_COUNT: usize = 10;
const BATCH_SIZE: usize = 1000;
const NUM_TASKS: usize = 4;

async fn benchmark_rstar(
    client: &Client,
    total_rows: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("--- R*-tree benchmark: {} rows ---", total_rows);

    let namespace_name = format!("rstar_benchmark_{}", uuid::Uuid::new_v4());
    client.create_namespace(&namespace_name, true).await?;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: BATCH_SIZE,
        parquet_row_group_size: 128,
        preferred_data_file_format: DataFileFormat::ParquetV2,
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: arrow_rstar_table_schema(),
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let task_batches = (total_rows / BATCH_SIZE - 1) / NUM_TASKS;
    let query_batch_size = total_rows - task_batches * NUM_TASKS * BATCH_SIZE;
    let query_batch = new_rstar_record_batch(query_batch_size);
    let query_wkbs: Vec<Vec<u8>> = {
        let geom = query_batch.column(1).as_binary_view();
        (0..QUERY_COUNT).map(|i| geom.value(i).to_vec()).collect()
    };

    let mut handles = Vec::with_capacity(NUM_TASKS);
    for _ in 0..NUM_TASKS {
        let table = table.clone();
        handles.push(tokio::spawn(async move {
            let stream = futures::stream::iter(
                (0..task_batches).map(|_| Ok::<_, ILError>(new_rstar_record_batch(BATCH_SIZE))),
            );
            table.bypass_insert(Box::pin(stream)).await
        }));
    }

    let start_time = Instant::now();
    let mut total_inserted = table
        .bypass_insert(Box::pin(futures::stream::iter([Ok::<_, ILError>(
            query_batch,
        )])))
        .await?;

    for handle in handles {
        total_inserted += handle.await??;
    }

    let insert_time = start_time.elapsed();
    println!(
        "insert: {} rows, {} tasks, batch size: {}, in {}ms",
        total_inserted,
        NUM_TASKS,
        BATCH_SIZE,
        insert_time.as_millis(),
    );

    let index_creation = IndexCreation {
        name: "rstar_index".to_string(),
        kind: RStarIndexKind.kind().to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        if_not_exists: false,
    };
    let table = client.load_table(&namespace_name, &table_name).await?;
    let index_start = Instant::now();
    table.create_index(index_creation).await?;
    let index_time = index_start.elapsed();
    println!("index build: {}ms", index_time.as_millis());

    let table = client.load_table(&namespace_name, &table_name).await?;

    let mut total_query_time = std::time::Duration::ZERO;
    let mut total_results = 0usize;

    for wkb in &query_wkbs {
        let scan = TableScan::default()
            .with_filters(vec![func(
                "intersects",
                vec![col("geom"), lit(wkb.clone())],
                DataType::Boolean,
            )])
            .with_batch_size(1000usize);

        let query_start = Instant::now();
        let mut stream = table.scan(scan).await?;
        let mut count = 0usize;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            count += batch.num_rows();
        }
        total_query_time += query_start.elapsed();
        total_results += count;
    }

    let avg_query_ms = total_query_time.as_millis() / QUERY_COUNT as u128;
    println!(
        "query: {} queries, avg {}ms, total {} results\n",
        QUERY_COUNT, avg_query_ms, total_results,
    );

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let catalog = catalog_postgres().await;
    let storage = storage_s3().await;

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RStarIndexKind));

    println!("=== IndexLake R*-tree benchmark suite ===\n");

    for total_rows in [100_000, 1_000_000] {
        benchmark_rstar(&client, total_rows).await?;
    }

    println!("=== benchmark complete ===");

    Ok(())
}
