use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableSearch};
use indexlake::{Client, ILError};
use indexlake_benchmarks::{benchprintln, wait_data_files_ready};
use indexlake_benchmarks::data::{arrow_vector_table_schema, new_vector_record_batch};
use indexlake_index_rabitq::{RabitqIndexKind, RabitqIndexParams, RabitqMetric, RabitqSearchQuery};
use indexlake_integration_tests::{catalog_postgres, init_env_logger, storage_s3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();
    let catalog = catalog_postgres().await;
    let storage = storage_s3().await;

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RabitqIndexKind));

    let namespace_name = "benchmark_rabitq";
    client.create_namespace(namespace_name, true).await?;

    let total_rows = 100000;
    let num_tasks = 10;
    let task_rows = total_rows / num_tasks;
    let insert_batch_size = 1000;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: 10000,
        parquet_row_group_size: 100,
        preferred_data_file_format: DataFileFormat::ParquetV2,
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.to_string(),
        table_name: table_name.clone(),
        schema: arrow_vector_table_schema(),
        default_values: HashMap::new(),
        config: table_config.clone(),
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let index_name = "rabitq_index";
    let index_creation = IndexCreation {
        name: index_name.to_string(),
        kind: RabitqIndexKind.kind().to_string(),
        key_columns: vec!["vector".to_string()],
        params: Arc::new(RabitqIndexParams {
            metric: RabitqMetric::L2,
            total_bits: 7,
        }),
        concurrency: 1,
        if_not_exists: false,
    };
    let table = client.load_table(namespace_name, &table_name).await?;
    table.create_index(index_creation).await?;

    let table = client.load_table(namespace_name, &table_name).await?;

    let start_time = Instant::now();
    let mut handles = Vec::new();
    for _ in 0..num_tasks {
        let table = table.clone();
        let handle = tokio::spawn(async move {
            let mut progress = 0;
            while progress < task_rows {
                let batch = new_vector_record_batch(insert_batch_size);
                table
                    .bypass_insert(Box::pin(futures::stream::iter(vec![Ok(batch)])))
                    .await?;
                progress += insert_batch_size;
            }
            Ok::<_, ILError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    let insert_elapsed = start_time.elapsed();
    benchprintln!(
        "IndexLake RaBitQ: inserted {total_rows} rows, {num_tasks} tasks, batch size: {insert_batch_size}, format: {}, in {}ms",
        table_config.preferred_data_file_format,
        insert_elapsed.as_millis()
    );

    wait_data_files_ready(
        &table,
        total_rows / table.config.inline_row_count_limit,
        Duration::from_secs(2000),
    )
    .await?;

    let start_time = Instant::now();
    let limit = 10;
    let table_search = TableSearch {
        query: Arc::new(RabitqSearchQuery {
            vector: vec![500.0; 1024],
            limit,
        }),
        projection: None,
        dynamic_fields: vec![],
    };
    let mut stream = table.search(table_search).await?;
    let mut search_count = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        search_count += batch.num_rows();
    }
    assert_eq!(search_count, limit);

    let search_elapsed = start_time.elapsed();
    benchprintln!(
        "IndexLake RaBitQ: searched {search_count} rows in {}ms",
        search_elapsed.as_millis()
    );

    Ok(())
}
