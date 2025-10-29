use std::collections::HashMap;
use std::time::Instant;

use futures::StreamExt;
use indexlake::expr::{col, lit};
use indexlake::table::{TableCreation, TableInsertion, TableScan, TableScanPartition, TableUpdate};
use indexlake::{Client, ILError};
use indexlake_benchmarks::data::{arrow_table_schema, new_record_batch};
use indexlake_integration_tests::{catalog_postgres, init_env_logger, storage_s3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();
    let catalog = catalog_postgres().await;
    let storage = storage_s3().await;

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: arrow_table_schema(),
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let total_rows = 1000000;
    let num_tasks = 10;
    let task_rows = total_rows / num_tasks;
    let insert_batch_size = 100000;

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
            Ok::<_, ILError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    let insert_cost_time = start_time.elapsed();
    println!(
        "IndexLake: inserted {} rows, {} tasks, batch size: {}, in {}ms",
        total_rows,
        num_tasks,
        insert_batch_size,
        insert_cost_time.as_millis()
    );

    let table_count = table.count(TableScanPartition::single_partition()).await?;
    assert_eq!(table_count, total_rows);

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
            Ok::<_, ILError>(count)
        });
        handles.push(handle);
    }
    let mut count = 0;
    for handle in handles {
        count += handle.await??;
    }

    assert_eq!(count, total_rows);

    let scan_cost_time = start_time.elapsed();
    println!(
        "IndexLake: scanned {} rows by {} tasks in {}ms",
        count,
        num_tasks,
        scan_cost_time.as_millis()
    );

    let start_time = Instant::now();
    let update = TableUpdate {
        set_map: HashMap::from([("content".to_string(), lit("new content"))]),
        condition: col("id").eq(lit(100i32)),
    };
    let update_count = table.update(update).await?;
    let update_cost_time = start_time.elapsed().as_millis();
    println!("IndexLake: updated {update_count} rows in {update_cost_time}ms",);

    let start_time = Instant::now();
    let condition = col("id").eq(lit(100i32));
    let delete_count = table.delete(condition).await?;
    let delete_cost_time = start_time.elapsed().as_millis();
    println!("IndexLake: deleted {delete_count} rows in {delete_cost_time}ms",);

    Ok(())
}
