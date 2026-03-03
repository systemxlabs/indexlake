use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{Int32Builder, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use indexlake::Client;
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableInsertion};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};
use indexlake_integration_tests::{catalog_sqlite, storage_fs};
use serde::Serialize;

#[derive(Debug, Serialize)]
struct BenchResult {
    rows: usize,
    batch_size: usize,
    inline_row_count_limit: usize,
    data_file_count: usize,
    concurrency: usize,
    insert_ms: u128,
    index_build_ms: u128,
}

fn parse_arg_usize(name: &str, default: usize) -> usize {
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == name
            && let Some(v) = args.next()
        {
            return v.parse::<usize>().unwrap_or(default);
        }
    }
    default
}

fn clean_fs_storage_dir_best_effort() {
    // integration-tests::storage_fs() uses: integration-tests/tmp/fs_storage
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let dir = root.join("../integration-tests/tmp/fs_storage");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // CLI:
    //   --concurrency N (required, must be >= 1)
    //   --rows N
    //   --batch-size N
    //   --inline-row-count-limit N
    let concurrency = parse_arg_usize("--concurrency", 1);
    if concurrency == 0 {
        eprintln!("Invalid --concurrency: must be >= 1");
        std::process::exit(2);
    }

    let rows = parse_arg_usize("--rows", 50_000);
    let batch_size = parse_arg_usize("--batch-size", 1_000);
    let inline_row_count_limit = parse_arg_usize("--inline-row-count-limit", 1_000);

    clean_fs_storage_dir_best_effort();

    let catalog = catalog_sqlite();
    let storage = storage_fs();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("integer", DataType::Int32, false),
    ]));

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
            schema: table_schema.clone(),
            config: table_config,
            ..Default::default()
        })
        .await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let insert_start = Instant::now();
    let mut next_id: i32 = 0;
    while (next_id as usize) < rows {
        let remaining = rows - (next_id as usize);
        let n = remaining.min(batch_size);

        let mut id_builder = Int32Builder::with_capacity(n);
        let mut int_builder = Int32Builder::with_capacity(n);
        for _ in 0..n {
            id_builder.append_value(next_id);
            int_builder.append_value(next_id % 10_000);
            next_id += 1;
        }

        let batch = RecordBatch::try_new(
            table_schema.clone(),
            vec![
                Arc::new(id_builder.finish()),
                Arc::new(int_builder.finish()),
            ],
        )?;
        table.insert(TableInsertion::new(vec![batch])).await?;
    }
    let insert_ms = insert_start.elapsed().as_millis();

    // Give background dump tasks time to materialize data files.
    tokio::time::sleep(Duration::from_secs(3)).await;

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

    let out = BenchResult {
        rows,
        batch_size,
        inline_row_count_limit,
        data_file_count,
        concurrency,
        insert_ms,
        index_build_ms,
    };

    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
