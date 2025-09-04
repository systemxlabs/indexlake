use std::time::Instant;

use arrow::array::RecordBatchIterator;
use arrow::record_batch::RecordBatch;
use arrow::util::pretty::pretty_format_batches;
use futures::StreamExt;
use indexlake_benchmarks::data::{arrow_bm25_table_schema, new_bm25_record_batch};
use indexlake_benchmarks::{set_lance_language_model_home, setup_s3_env};
use indexlake_integration_tests::setup_minio;
use lance::Dataset;
use lance_index::optimize::OptimizeOptions;
use lance_index::scalar::{FullTextSearchQuery, InvertedIndexParams};
use lance_index::{DatasetIndexExt, IndexType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    setup_minio();
    setup_s3_env();
    set_lance_language_model_home();

    // create table
    let table_path = "s3://indexlake/lance/bm25_bench_table".to_string();
    let batch = RecordBatch::new_empty(arrow_bm25_table_schema());
    let batch_reader = RecordBatchIterator::new(vec![Ok(batch)], arrow_bm25_table_schema());
    let mut dataset = Dataset::write(batch_reader, &table_path, None).await?;

    let params = InvertedIndexParams::default().base_tokenizer("jieba".to_owned());
    dataset
        .create_index(
            &["content"],
            IndexType::Inverted,
            Some("bm25_index".to_owned()),
            &params,
            false,
        )
        .await?;

    // insert data
    let total_rows = 1000000;
    let insert_batch_size = 10000;

    let start_time = Instant::now();
    let mut progress = 0;
    while progress < total_rows {
        let batch = new_bm25_record_batch(insert_batch_size);
        let record_batch_reader =
            RecordBatchIterator::new(vec![batch].into_iter().map(Ok), arrow_bm25_table_schema());
        dataset.append(record_batch_reader, None).await?;
        dataset.optimize_indices(&OptimizeOptions::new()).await?;
        progress += insert_batch_size;
    }

    println!(
        "Lance: inserted {} bm25 rows ({} per batch) in {}ms",
        total_rows,
        insert_batch_size,
        start_time.elapsed().as_millis()
    );

    // search data
    let start_time = Instant::now();
    let dataset = Dataset::open(&table_path).await?;
    let limit = 10;
    let query = FullTextSearchQuery::new("杨绛女士".to_owned())
        .limit(Some(limit))
        .with_column("content".to_owned())?;
    let mut scanner = dataset.scan();
    scanner.full_text_search(query)?;
    let mut stream = scanner.try_into_stream().await?;
    let mut batches = vec![];
    while let Some(batch) = stream.next().await {
        let batch = batch?;
        batches.push(batch);
    }

    println!(
        "Lance: searched {} bm25 rows in {}ms",
        limit,
        start_time.elapsed().as_millis()
    );

    println!("{}", pretty_format_batches(&batches).unwrap());
    Ok(())
}
