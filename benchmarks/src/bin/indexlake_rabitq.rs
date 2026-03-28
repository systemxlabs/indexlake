use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableSearch};
use indexlake::{Client, ILError};
use indexlake_benchmarks::benchprintln;
use indexlake_benchmarks::data::{arrow_vector_table_schema, new_vector_record_batch};
use indexlake_index_rabitq::{
    RabitqAlgo, RabitqIndexKind, RabitqIndexParams, RabitqMetric, RabitqSearchQuery,
};
use indexlake_integration_tests::{catalog_postgres, init_env_logger, storage_s3};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();
    let namespace_name = "bench_rabitq";

    bench_algo(
        namespace_name,
        RabitqIndexParams {
            algo: RabitqAlgo::BruteForce,
            metric: RabitqMetric::L2,
            total_bits: 7,
            nlist: 256,
        },
    )
    .await?;

    bench_algo(
        namespace_name,
        RabitqIndexParams {
            algo: RabitqAlgo::Ivf,
            metric: RabitqMetric::L2,
            total_bits: 7,
            nlist: 64,
        },
    )
    .await?;

    Ok(())
}

async fn bench_algo(
    namespace_name: &str,
    params: RabitqIndexParams,
) -> Result<(), Box<dyn std::error::Error>> {
    let catalog = catalog_postgres().await;
    let storage = storage_s3().await;

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RabitqIndexKind));

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

    let algo_name = format!("{:?}", params.algo);
    let index_name = format!("rabitq_{}_index", algo_name.to_lowercase());
    let index_creation = IndexCreation {
        name: index_name.clone(),
        kind: RabitqIndexKind.kind().to_string(),
        key_columns: vec!["vector".to_string()],
        params: Arc::new(params),
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
        "RaBitQ {algo_name}: inserted {total_rows} rows, {num_tasks} tasks, batch size: {insert_batch_size}, format: {}, in {}ms",
        table_config.preferred_data_file_format,
        insert_elapsed.as_millis()
    );

    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    let search_query = RabitqSearchQuery {
        vector: vec![500.0f32; 1024],
        limit: 10,
        nprobe: 8,
    };

    let start_time = Instant::now();
    let table_search = TableSearch {
        query: Arc::new(search_query),
        projection: None,
        dynamic_fields: vec![],
    };
    let mut stream = table.search(table_search).await?;
    let mut search_count = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        search_count += batch.num_rows();
    }
    let search_elapsed = start_time.elapsed();
    benchprintln!(
        "RaBitQ {algo_name}: searched {search_count} rows in {}ms",
        search_elapsed.as_millis()
    );

    Ok(())
}
