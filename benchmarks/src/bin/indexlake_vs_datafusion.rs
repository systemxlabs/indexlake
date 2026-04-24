use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::prelude::SessionContext;
use datafusion::sql::TableReference;
use futures::StreamExt;
use indexlake::Client;
use indexlake::table::{TableConfig, TableCreation, TableInsertion, TableScan, TableScanPartition};
use indexlake_benchmarks::data::{arrow_table_schema, new_record_batch};
use indexlake_benchmarks::{benchprintln, wait_data_files_ready};
use indexlake_integration_tests::{catalog_postgres, init_env_logger, storage_s3};
use object_store::aws::AmazonS3Builder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    // Setup shared resources
    let catalog = catalog_postgres().await;
    let storage = storage_s3().await;

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: 10000,
        ..Default::default()
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: arrow_table_schema(),
        default_values: HashMap::new(),
        config: table_config.clone(),
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    // Insert data
    let total_rows = 1_000_000usize;
    let num_tasks = 10usize;
    let task_rows = total_rows / num_tasks;
    let insert_batch_size = 10_000usize;

    let start_time = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..num_tasks {
        let table = table.clone();
        let handle = tokio::spawn(async move {
            let mut progress = 0;
            while progress < task_rows {
                let batch = new_record_batch(insert_batch_size);
                table.insert(TableInsertion::new(vec![batch])).await?;
                progress += insert_batch_size;
            }
            Ok::<_, indexlake::ILError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    let insert_cost_time = start_time.elapsed();
    benchprintln!(
        "IndexLake: inserted {} rows, {} tasks, batch size: {}, in {}ms",
        total_rows,
        num_tasks,
        insert_batch_size,
        insert_cost_time.as_millis()
    );

    wait_data_files_ready(
        &table,
        total_rows / table.config.inline_row_count_limit,
        Duration::from_secs(300),
    )
    .await?;

    let listing_table_path = format!("s3://indexlake/{}/{}", table.namespace_id, table.table_id);

    // Run IndexLake full table scan benchmark
    bench_indexlake_scan(&table, total_rows, num_tasks).await?;

    // Run DataFusion listing table full scan benchmark
    bench_datafusion_scan(&listing_table_path).await?;

    Ok(())
}

async fn bench_indexlake_scan(
    table: &indexlake::table::Table,
    total_rows: usize,
    num_tasks: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();
    let mut handles = Vec::new();

    for i in 0..num_tasks {
        let table = table.clone();
        let handle = tokio::spawn(async move {
            let scan = TableScan::default().with_partition(TableScanPartition::Auto {
                partition_idx: i,
                partition_count: num_tasks,
            });
            let mut stream = table.scan(scan).await?;
            let mut count = 0;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                count += batch.num_rows();
            }
            Ok::<_, indexlake::ILError>(count)
        });
        handles.push(handle);
    }

    let mut count = 0;
    for handle in handles {
        count += handle.await??;
    }

    assert_eq!(count, total_rows);

    let scan_cost_time = start_time.elapsed();
    benchprintln!(
        "IndexLake: scanned {} rows by {} tasks in {}ms",
        count,
        num_tasks,
        scan_cost_time.as_millis()
    );

    Ok(())
}

async fn bench_datafusion_scan(table_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Create S3 object store
    let s3 = AmazonS3Builder::new()
        .with_endpoint("http://127.0.0.1:9000")
        .with_bucket_name("indexlake")
        .with_region("us-east-1")
        .with_access_key_id("admin")
        .with_secret_access_key("password")
        .with_allow_http(true)
        .build()?;

    // Create session context with object store
    let ctx = SessionContext::new();
    let object_store_url = url::Url::parse("s3://indexlake/")?;
    ctx.register_object_store(&object_store_url, Arc::new(s3.clone()));

    // Define schema (same as indexlake table)
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, true),
        Field::new("content", DataType::Utf8, true),
        Field::new("data", DataType::Binary, true),
    ]));

    // Create listing table config
    let table_url = ListingTableUrl::parse(format!("{}/", table_path))?;
    let listing_options = ListingOptions::new(Arc::new(
        datafusion::datasource::file_format::parquet::ParquetFormat::default(),
    ))
    .with_file_extension(".parquet");

    let config = ListingTableConfig::new(table_url)
        .with_listing_options(listing_options)
        .with_schema(schema);

    let listing_table = ListingTable::try_new(config)?;

    // Register table and run query
    ctx.register_table(
        TableReference::bare("listing_table"),
        Arc::new(listing_table),
    )?;

    let df = ctx.sql("SELECT * FROM listing_table").await?;

    let start_time = Instant::now();

    let batches = df.collect().await?;

    let count: usize = batches.iter().map(|b| b.num_rows()).sum();

    let scan_cost_time = start_time.elapsed();
    benchprintln!(
        "DataFusion listing table: scanned {} rows in {}ms",
        count,
        scan_cost_time.as_millis()
    );

    Ok(())
}
