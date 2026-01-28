use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use futures::StreamExt;
use indexlake::Client;
use indexlake::expr::{col, lit};
use indexlake::table::{Table, TableCreation, TableInsertion, TableScan, TableUpdate};
use indexlake_catalog_sqlite::SqliteCatalog;
use indexlake_storage_fs::FsStorage;
use rusqlite::Connection;

const SQLITE_CATALOG_SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS indexlake_namespace (
    namespace_id BLOB PRIMARY KEY,
    namespace_name VARCHAR NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS indexlake_table (
    table_id BLOB PRIMARY KEY,
    table_name VARCHAR NOT NULL,
    namespace_id BLOB NOT NULL,
    config VARCHAR NOT NULL,
    schema_metadata VARCHAR NOT NULL,
    UNIQUE (namespace_id, table_name)
);

CREATE TABLE IF NOT EXISTS indexlake_field (
    field_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    field_name VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    nullable BOOLEAN NOT NULL,
    default_value VARCHAR,
    metadata VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_task (
    task_id VARCHAR PRIMARY KEY,
    start_at BIGINT NOT NULL,
    max_lifetime BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_data_file (
    data_file_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    format VARCHAR NOT NULL,
    relative_path VARCHAR NOT NULL,
    size BIGINT NOT NULL,
    record_count BIGINT NOT NULL,
    validity BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_index (
    index_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    index_name VARCHAR NOT NULL,
    index_kind VARCHAR NOT NULL,
    key_field_ids VARCHAR NOT NULL,
    params VARCHAR NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_index_file (
    index_file_id BLOB PRIMARY KEY,
    table_id BLOB NOT NULL,
    index_id BLOB NOT NULL,
    data_file_id BLOB NOT NULL,
    relative_path VARCHAR NOT NULL,
    size BIGINT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_inline_index (
    index_id BLOB NOT NULL,
    index_data BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS indexlake_data_file_stats (
    data_file_id BLOB NOT NULL,
    field_id BLOB NOT NULL,
    min_value VARCHAR,
    max_value VARCHAR
);
"#;

fn init_sqlite_catalog(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.execute_batch(SQLITE_CATALOG_SCHEMA)?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_dir = std::env::temp_dir().join("indexlake_basic_crud");
    let catalog_path = base_dir.join("catalog.db");
    init_sqlite_catalog(&catalog_path)?;

    let catalog = Arc::new(SqliteCatalog::try_new(catalog_path.to_string_lossy())?);
    let storage = Arc::new(FsStorage::new(base_dir.join("data")));
    let client = Client::new(catalog, storage);

    client.create_namespace("default", true).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));

    let table_creation = TableCreation {
        namespace_name: "default".to_string(),
        table_name: "users".to_string(),
        schema: schema.clone(),
        if_not_exists: true,
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table("default", "users").await?;

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(Int32Array::from(vec![20, 21, 22])),
        ],
    )?;
    table.insert(TableInsertion::new(vec![batch])).await?;

    let batches = collect_scan(
        &table,
        TableScan::default().with_projection(Some(vec![1, 2])),
    )
    .await?;
    println!("after insert:");
    print_users(&batches)?;

    let update = TableUpdate {
        set_map: HashMap::from([("age".to_string(), lit(30i32))]),
        condition: col("name").eq(lit("Alice")),
    };
    let updated = table.update(update).await?;
    println!("updated rows: {updated}");

    let deleted = table.delete(col("age").gt(lit(25i32))).await?;
    println!("deleted rows: {deleted}");

    let batches = collect_scan(
        &table,
        TableScan::default().with_projection(Some(vec![1, 2])),
    )
    .await?;
    println!("after update/delete:");
    print_users(&batches)?;

    Ok(())
}

async fn collect_scan(
    table: &Table,
    scan: TableScan,
) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
    let mut stream = table.scan(scan).await?;
    let mut batches = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch?);
    }
    Ok(batches)
}

fn print_users(batches: &[RecordBatch]) -> Result<(), Box<dyn std::error::Error>> {
    for batch in batches {
        let names = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or("expected StringArray in column 0")?;
        let ages = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or("expected Int32Array in column 1")?;
        for row in 0..batch.num_rows() {
            let name = if names.is_null(row) {
                "NULL".to_string()
            } else {
                names.value(row).to_string()
            };
            let age = if ages.is_null(row) {
                "NULL".to_string()
            } else {
                ages.value(row).to_string()
            };
            println!("{name} | {age}");
        }
    }
    Ok(())
}
