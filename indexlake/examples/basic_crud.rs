use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use futures::StreamExt;
use indexlake::Client;
use indexlake::expr::{col, lit};
use indexlake::table::{Table, TableCreation, TableInsertion, TableScan, TableUpdate};
use indexlake_catalog_postgres::PostgresCatalogBuilder;
use indexlake_storage_s3::S3Storage;
use opendal::services::S3Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let catalog = Arc::new(
        PostgresCatalogBuilder::new("fake-postgres", 5432, "postgres", "password")
            .dbname("postgres")
            .pool_max_size(5)
            .pool_idle_timeout(Some(Duration::from_secs(10)))
            .build()
            .await?,
    );

    let mut config = S3Config::default();
    config.endpoint = Some("http://fake-s3:9000".to_string());
    config.access_key_id = Some("fake-access".to_string());
    config.secret_access_key = Some("fake-secret".to_string());
    config.region = Some("us-east-1".to_string());
    config.disable_config_load = true;
    config.disable_ec2_metadata = true;

    let storage = Arc::new(S3Storage::new(config, "fake-bucket".into()));
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

    let _batches = collect_scan(
        &table,
        TableScan::default().with_projection(Some(vec![1, 2])),
    )
    .await?;

    let update = TableUpdate {
        set_map: HashMap::from([("age".to_string(), lit(30i32))]),
        condition: col("name").eq(lit("Alice")),
    };
    let updated = table.update(update).await?;
    println!("updated rows: {updated}");

    let deleted = table.delete(col("age").gt(lit(25i32))).await?;
    println!("deleted rows: {deleted}");

    let _batches = collect_scan(
        &table,
        TableScan::default().with_projection(Some(vec![1, 2])),
    )
    .await?;

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
