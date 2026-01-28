use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use indexlake::Client;
use indexlake::expr::{col, lit};
use indexlake::table::{TableCreation, TableInsertion, TableScan, TableUpdate};
use indexlake_catalog_postgres::PostgresCatalogBuilder;
use indexlake_storage_s3::S3Storage;
use opendal::services::S3Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fake Postgres connection settings for documentation/demo purposes.
    let catalog = Arc::new(
        PostgresCatalogBuilder::new("fake-postgres", 5432, "postgres", "password")
            .dbname("postgres")
            .build()
            .await?,
    );

    // Fake S3 settings; this example is not expected to connect successfully.
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

    // Scan once (project name + age, skipping the internal row id).
    let _scan_stream = table
        .scan(TableScan::default().with_projection(Some(vec![1, 2])))
        .await?;

    let update = TableUpdate {
        set_map: HashMap::from([("age".to_string(), lit(30i32))]),
        condition: col("name").eq(lit("Alice")),
    };
    let updated = table.update(update).await?;
    println!("updated rows: {updated}");

    let deleted = table.delete(col("age").gt(lit(25i32))).await?;
    println!("deleted rows: {deleted}");

    Ok(())
}
