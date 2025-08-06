use std::time::Instant;

use futures::StreamExt;
use indexlake::ILError;
use indexlake::table::TableConfig;
use indexlake::table::TableCreation;
use indexlake::table::TableScan;
use indexlake::table::TableScanPartition;
use indexlake::{Client, storage::DataFileFormat};
use indexlake_benchmarks::data::{arrow_table_schema, new_record_batch};
use indexlake_integration_tests::init_env_logger;
use indexlake_integration_tests::{catalog_postgres, storage_s3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();
    let catalog = catalog_postgres().await;
    let storage = storage_s3();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: 10000,
        parquet_row_group_size: 1000,
        preferred_data_file_format: DataFileFormat::LanceV2_0,
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: arrow_table_schema(),
        config: table_config,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let total_rows = 1000000;
    let num_tasks = 10;
    let task_rows = total_rows / num_tasks;
    let insert_batch_size = 10000;

    let start_time = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..num_tasks {
        let table = table.clone();
        let handle = tokio::spawn(async move {
            let mut progress = 0;
            while progress < task_rows {
                let batch = new_record_batch(insert_batch_size);
                table.insert(&[batch]).await?;
                progress += insert_batch_size;
            }
            Ok::<_, ILError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    let insert_cost_time = start_time.elapsed();
    println!(
        "IndexLake: inserted {} rows by {} tasks in {}ms",
        total_rows,
        num_tasks,
        insert_cost_time.as_millis()
    );

    let start_time = Instant::now();
    let mut handles = Vec::new();
    for i in 0..num_tasks {
        let table = table.clone();
        let handle = tokio::spawn(async move {
            let scan = TableScan::default().with_partition(TableScanPartition {
                partition_idx: i,
                partition_count: num_tasks,
            });
            let mut stream = table.scan(scan).await?;
            let mut count = 0;
            while let Some(batch) = stream.next().await {
                let batch = batch?;
                count += batch.num_rows();
            }
            Ok::<_, ILError>(count)
        });
        handles.push(handle);
    }
    let mut count = 0;
    for handle in handles {
        count += handle.await??;
    }

    let scan_cost_time = start_time.elapsed();
    println!(
        "IndexLake: scanned {} rows by {} tasks in {}ms",
        count,
        num_tasks,
        scan_cost_time.as_millis()
    );

    Ok(())
}
