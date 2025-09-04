use std::collections::HashMap;
use std::time::Instant;

use delta_kernel::engine::arrow_conversion::TryFromArrow;
use deltalake::operations::create::CreateBuilder;
use deltalake::protocol::SaveMode;
use deltalake::{DeltaOps, StructField, open_table_with_storage_options};
use futures::StreamExt;
use indexlake_benchmarks::data::{arrow_table_schema, new_record_batch};
use indexlake_integration_tests::setup_minio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_minio();

    deltalake_aws::register_handlers(None);

    // create table
    let table_path = "s3://indexlake/deltalake/bench_table";
    let _ = CreateBuilder::new()
        .with_storage_options(deltalake_storage_options())
        .with_table_name("bench_table")
        .with_columns(delta_table_schema())
        .with_location(table_path)
        .with_save_mode(SaveMode::ErrorIfExists)
        .await?;

    // insert data
    let total_rows = 1000000;
    let insert_batch_size = 10000;

    let start_time = Instant::now();
    let mut progress = 0;
    while progress < total_rows {
        let table =
            open_table_with_storage_options(&table_path, deltalake_storage_options()).await?;
        let ops = DeltaOps(table);
        let batch = new_record_batch(insert_batch_size);
        ops.write(vec![batch]).await?;
        progress += insert_batch_size;
    }

    println!(
        "DeltaLake: Inserted {} rows ({} per batch) in {}ms",
        total_rows,
        insert_batch_size,
        start_time.elapsed().as_millis()
    );

    // scan data
    let table = open_table_with_storage_options(&table_path, deltalake_storage_options()).await?;
    let start_time = Instant::now();
    let (_table, mut stream) = DeltaOps(table).load().await?;
    let mut count = 0;
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        count += batch.num_rows();
    }
    println!(
        "DeltaLake: Scanned {} rows in {}ms",
        count,
        start_time.elapsed().as_millis()
    );
    Ok(())
}

fn deltalake_storage_options() -> HashMap<String, String> {
    let mut options = HashMap::new();
    options.insert("aws_access_key_id".to_string(), "admin".to_string());
    options.insert("aws_secret_access_key".to_string(), "password".to_string());
    options.insert(
        "aws_endpoint".to_string(),
        "http://127.0.0.1:9000".to_string(),
    );
    options.insert("aws_allow_http".to_string(), "true".to_string());
    options.insert("aws_default_region".to_string(), "us-east-1".to_string());
    options.insert("AWS_EC2_METADATA_DISABLED".to_string(), "true".to_string());
    options.insert("AWS_S3_ALLOW_UNSAFE_RENAME".to_string(), "true".to_string());
    options
}

fn delta_table_schema() -> Vec<StructField> {
    let mut fields: Vec<StructField> = vec![];
    for arrow_field in arrow_table_schema().fields() {
        fields.push(StructField::try_from_arrow(arrow_field).unwrap());
    }
    fields
}
