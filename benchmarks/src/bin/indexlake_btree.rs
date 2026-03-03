use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use futures::StreamExt;
use indexlake::catalog::Scalar;
use indexlake::expr::{col, lit};
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableInsertion, TableScan};
use indexlake::{Client, ILError};
use indexlake_benchmarks::data::{
    arrow_btree_integer_table_schema, arrow_btree_string_table_schema,
    new_btree_integer_record_batch, new_btree_string_record_batch,
};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};

fn has_flag(flag: &str) -> bool {
    std::env::args().any(|a| a == flag)
}

fn clean_fs_storage_dir_best_effort() {
    // integration-tests::storage_fs() uses: integration-tests/tmp/fs_storage
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../integration-tests/tmp/fs_storage");
    let _ = std::fs::remove_dir_all(&dir);
}

async fn run_ci() -> Result<(), Box<dyn std::error::Error>> {
    // Fixed parameters for CI repeatability.
    let rows: usize = 50_000;
    let batch_size: usize = 1_000;
    let inline_row_count_limit: usize = 5_000;
    let concurrency: usize = 1;

    clean_fs_storage_dir_best_effort();

    let catalog = catalog_sqlite();
    let storage = storage_fs();
    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit,
        parquet_row_group_size: 1024,
        preferred_data_file_format: DataFileFormat::ParquetV2,
    };
    client
        .create_table(TableCreation {
            namespace_name: namespace_name.clone(),
            table_name: table_name.clone(),
            schema: arrow_btree_integer_table_schema(),
            default_values: HashMap::new(),
            config: table_config,
            if_not_exists: false,
        })
        .await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let insert_start = Instant::now();
    let mut inserted = 0usize;
    while inserted < rows {
        let n = (rows - inserted).min(batch_size);
        table
            .insert(TableInsertion::new(vec![new_btree_integer_record_batch(n)]))
            .await?;
        inserted += n;
    }
    let insert_ms = insert_start.elapsed().as_millis();

    // Wait for background dump task to finish so index build sees data files.
    let expected_files = rows / inline_row_count_limit;
    let dump_deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let inline_rows = table.inline_row_count().await?;
        let data_files = table.data_file_count().await?;
        if inline_rows < inline_row_count_limit && data_files >= expected_files {
            break;
        }
        if Instant::now() > dump_deadline {
            return Err(format!(
                "dump did not finish in time (inline_rows={inline_rows}, data_files={data_files}, expected_files={expected_files})"
            )
            .into());
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    let data_file_count = table.data_file_count().await?;

    let index_creation = IndexCreation {
        name: "btree_ci_bench".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["integer".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        concurrency,
        if_not_exists: false,
    };

    let index_start = Instant::now();
    table.clone().create_index(index_creation).await?;
    let index_build_ms = index_start.elapsed().as_millis();

    // Print a single JSON line so CI can extract and comment it.
    println!(
        "{{\"rows\":{},\"batch_size\":{},\"inline_row_count_limit\":{},\"data_file_count\":{},\"concurrency\":{},\"insert_ms\":{},\"index_build_ms\":{}}}",
        rows,
        batch_size,
        inline_row_count_limit,
        data_file_count,
        concurrency,
        insert_ms,
        index_build_ms
    );

    Ok(())
}

#[derive(Clone)]
enum DataType {
    Integer,
    String,
}

#[derive(Clone)]
struct BenchmarkConfig {
    data_type: DataType,
    total_rows: usize,
    batch_size: usize,
    concurrent_tasks: usize,
    namespace_suffix: String,
}

impl BenchmarkConfig {
    fn new_integer(total_rows: usize) -> Self {
        Self {
            data_type: DataType::Integer,
            total_rows,
            batch_size: 10000,
            concurrent_tasks: 10,
            namespace_suffix: "integer".to_string(),
        }
    }

    fn new_string(total_rows: usize) -> Self {
        Self {
            data_type: DataType::String,
            total_rows,
            batch_size: 10000,
            concurrent_tasks: 5,
            namespace_suffix: "string".to_string(),
        }
    }

    fn namespace_name(&self) -> String {
        format!("btree_benchmark_{}", self.namespace_suffix)
    }

    fn schema(&self) -> SchemaRef {
        match self.data_type {
            DataType::Integer => arrow_btree_integer_table_schema(),
            DataType::String => arrow_btree_string_table_schema(),
        }
    }

    fn record_batch(&self, batch_size: usize) -> RecordBatch {
        match self.data_type {
            DataType::Integer => new_btree_integer_record_batch(batch_size),
            DataType::String => new_btree_string_record_batch(batch_size),
        }
    }

    fn key_column(&self) -> String {
        match self.data_type {
            DataType::Integer => "integer".to_owned(),
            DataType::String => "string".to_owned(),
        }
    }

    fn index_name(&self) -> String {
        format!("{}_index", self.key_column())
    }

    fn point_query(&self) -> TableScan {
        match self.data_type {
            DataType::Integer => TableScan::default()
                .with_filters(vec![col("integer").eq(lit(Scalar::Int32(Some(100))))])
                .with_batch_size(100usize),
            DataType::String => TableScan::default()
                .with_filters(vec![
                    col("string").eq(lit(Scalar::Utf8(Some("Alice000100".to_string())))),
                ])
                .with_batch_size(100usize),
        }
    }

    fn range_query(&self) -> TableScan {
        match self.data_type {
            DataType::Integer => TableScan::default()
                .with_filters(vec![
                    col("integer").gteq(lit(Scalar::Int32(Some(100)))),
                    col("integer").lteq(lit(Scalar::Int32(Some(200)))),
                ])
                .with_batch_size(100usize),
            DataType::String => TableScan::default()
                .with_filters(vec![
                    col("string").gteq(lit(Scalar::Utf8(Some("Alice".to_string())))),
                    col("string").lteq(lit(Scalar::Utf8(Some("Bob".to_string())))),
                ])
                .with_batch_size(100usize),
        }
    }
}

struct BenchmarkResult {
    config: BenchmarkConfig,
    index_build_time_ms: u128,
    point_query_time_ms: u128,
    range_query_time_ms: u128,
    point_results_count: usize,
    range_results_count: usize,
}

impl BenchmarkResult {
    fn print_summary(&self) {
        println!(
            "=== B-tree index performance ({}) | rows: {:>6} ===",
            self.config.key_column(),
            self.config.total_rows,
        );
        println!(
            "build: {:>6}ms | point: {:>4}ms ({:>3} results) | range: {:>4}ms ({:>4} results)",
            self.index_build_time_ms,
            self.point_query_time_ms,
            self.point_results_count,
            self.range_query_time_ms,
            self.range_results_count
        );
    }
}

struct BenchmarkContext {
    client: Client,
}

impl BenchmarkContext {
    async fn init() -> Result<Self, Box<dyn std::error::Error>> {
        let catalog = catalog_postgres().await;
        let storage = storage_s3().await;
        let mut client = Client::new(catalog, storage);
        client.register_index_kind(Arc::new(BTreeIndexKind));

        Ok(Self { client })
    }

    async fn create_table_and_index(
        &self,
        config: &BenchmarkConfig,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let namespace_name = config.namespace_name();
        self.client.create_namespace(&namespace_name, true).await?;

        let table_name = uuid::Uuid::new_v4().to_string();
        let table_config = TableConfig {
            inline_row_count_limit: config.batch_size,
            parquet_row_group_size: 128,
            preferred_data_file_format: DataFileFormat::ParquetV2,
        };
        let table_creation = TableCreation {
            namespace_name: namespace_name.clone(),
            table_name: table_name.clone(),
            schema: config.schema(),
            default_values: HashMap::new(),
            config: table_config,
            if_not_exists: false,
        };
        self.client.create_table(table_creation).await?;

        let index_creation = IndexCreation {
            name: config.index_name(),
            kind: BTreeIndexKind.kind().to_string(),
            key_columns: vec![config.key_column().to_string()],
            params: Arc::new(BTreeIndexParams {}),
            concurrency: 1,
            if_not_exists: false,
        };
        let table = self.client.load_table(&namespace_name, &table_name).await?;
        table.create_index(index_creation).await?;

        Ok(table_name)
    }

    async fn insert_data_concurrent(
        &self,
        config: &BenchmarkConfig,
        table_name: &str,
    ) -> Result<std::time::Duration, Box<dyn std::error::Error>> {
        let namespace_name = config.namespace_name();
        let table = self.client.load_table(&namespace_name, table_name).await?;

        let start_time = Instant::now();

        let task_rows = config.total_rows / config.concurrent_tasks;
        let mut handles = Vec::with_capacity(config.concurrent_tasks);

        for _ in 0..config.concurrent_tasks {
            let table = table.clone();
            let config = config.clone();
            let handle = tokio::spawn(async move {
                let mut progress = 0;
                while progress < task_rows {
                    let batch_size = (task_rows - progress).min(config.batch_size);
                    table
                        .insert(TableInsertion::new(vec![config.record_batch(batch_size)]))
                        .await?;
                    progress += batch_size;
                }
                Ok::<_, ILError>(())
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.await??;
        }

        Ok(start_time.elapsed())
    }

    async fn execute_query(
        &self,
        config: &BenchmarkConfig,
        table_name: &str,
        scan: TableScan,
    ) -> Result<(std::time::Duration, usize), Box<dyn std::error::Error>> {
        let namespace_name = config.namespace_name();
        let table = self.client.load_table(&namespace_name, table_name).await?;

        let start_time = Instant::now();

        let mut stream = table.scan(scan).await?;
        let mut batches = vec![];
        while let Some(batch) = stream.next().await {
            batches.push(batch?);
        }

        let query_time = start_time.elapsed();
        let results_count: usize = batches.iter().map(|b| b.num_rows()).sum();

        Ok((query_time, results_count))
    }

    async fn benchmark_btree(
        &self,
        config: BenchmarkConfig,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let table_name = self.create_table_and_index(&config).await?;

        let insert_time = self.insert_data_concurrent(&config, &table_name).await?;

        tokio::time::sleep(tokio::time::Duration::from_secs(20)).await;

        let point_query = config.point_query();

        let (point_query_time, point_results_count) = self
            .execute_query(&config, &table_name, point_query)
            .await?;

        let range_query = config.range_query();

        let (range_query_time, range_results_count) = self
            .execute_query(&config, &table_name, range_query)
            .await?;

        Ok(BenchmarkResult {
            config,
            index_build_time_ms: insert_time.as_millis(),
            point_query_time_ms: point_query_time.as_millis(),
            range_query_time_ms: range_query_time.as_millis(),
            point_results_count,
            range_results_count,
        })
    }

    async fn benchmark_btree_integer(
        &self,
        total_rows: usize,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let config = BenchmarkConfig::new_integer(total_rows);
        self.benchmark_btree(config).await
    }

    async fn benchmark_btree_string(
        &self,
        total_rows: usize,
    ) -> Result<BenchmarkResult, Box<dyn std::error::Error>> {
        let config = BenchmarkConfig::new_string(total_rows);
        self.benchmark_btree(config).await
    }
}

async fn run_data_type_comparison() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== data type comparison benchmark ===");

    let context = BenchmarkContext::init().await?;

    println!("running integer benchmark (100,000 rows)...");
    let int_result = context.benchmark_btree_integer(100000).await?;
    int_result.print_summary();

    println!("running string benchmark (100,000 rows)...");
    let string_result = context.benchmark_btree_string(100000).await?;
    string_result.print_summary();

    print_data_type_comparison_summary(&int_result, &string_result);
    Ok(())
}

fn print_data_type_comparison_summary(
    int_result: &BenchmarkResult,
    string_result: &BenchmarkResult,
) {
    println!("=== performance comparison ===");
    println!(
        "{:>8} | {:>7} | {:>9} | {:>9} | {:>9}",
        "type", "rows", "build(ms)", "point(ms)", "range(ms)"
    );
    println!("{}", "-".repeat(54));

    for result in [int_result, string_result] {
        println!(
            "{:>8} | {:>7} | {:>9} | {:>9} | {:>9}",
            result.config.key_column(),
            result.config.total_rows,
            result.index_build_time_ms,
            result.point_query_time_ms,
            result.range_query_time_ms
        );
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    if has_flag("--ci") {
        return run_ci().await;
    }

    init_env_logger();

    println!("=== IndexLake B-tree benchmark suite: data types ===");
    run_data_type_comparison().await?;
    println!("=== benchmark complete ===");

    Ok(())
}
