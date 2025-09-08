use arrow::array::{AsArray, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Int64Type, Schema};
use futures::TryStreamExt;
use indexlake::catalog::{Catalog, Scalar};
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{TableConfig, TableCreation, TableInsertion, TableScan};
use indexlake::{Client, ILError};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::full_table_scan;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

#[rstest::rstest]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn parallel_insert_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: 100,
        parquet_row_group_size: 10,
        preferred_data_file_format: format,
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let data = (0..1000i64).collect::<Vec<_>>();
    let data_chunks = data.chunks(100);

    let mut handles = Vec::new();
    for (idx, data_chunk) in data_chunks.enumerate() {
        let table = table.clone();
        let table_schema = table_schema.clone();
        let data_chunk = data_chunk.to_vec();

        let handle = tokio::spawn(async move {
            let chunks = data_chunk.chunks(idx + 1);
            for chunk in chunks {
                let record_batch = RecordBatch::try_new(
                    table_schema.clone(),
                    vec![Arc::new(Int64Array::from(chunk.to_vec()))],
                )?;
                table
                    .insert(TableInsertion::new(vec![record_batch]))
                    .await?;
            }
            Ok::<(), ILError>(())
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.await??;
    }

    tokio::time::sleep(Duration::from_secs(5)).await;

    let stream = table.scan(TableScan::default()).await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    let mut read_data = Vec::new();
    for batch in batches {
        let batch_data = batch.column(1).as_primitive::<Int64Type>();
        read_data.extend(batch_data.iter().map(|v| v.unwrap()));
    }
    read_data.sort();
    assert_eq!(read_data, data);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn bypass_insert_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![Arc::new(Int64Array::from_iter_values(0..3))],
    )?;
    table.insert(TableInsertion::new(vec![batch])).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");
    assert_eq!(
        table_str,
        r#"+----+
| id |
+----+
| 0  |
| 1  |
| 2  |
+----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn insert_string_with_quotes(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![Arc::new(StringArray::from_iter_values(vec![
            "'A'", "'B", "''C''", "''D'", "A'B'C", r#""E""#, r#""F"#, r#"""G"""#, r#"""H""#,
        ]))],
    )?;
    table.insert(TableInsertion::new(vec![batch])).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");
    assert_eq!(
        table_str,
        r#"+-------+
| name  |
+-------+
| 'A'   |
| 'B    |
| ''C'' |
| ''D'  |
| A'B'C |
| "E"   |
| "F    |
| ""G"" |
| ""H"  |
+-------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn partial_insert_with_default_values(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let default_values = HashMap::from([("age".to_string(), Scalar::from(24i32))]);
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        default_values,
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let batch = RecordBatch::try_new(
        Arc::new(table_schema.project(&[0])?),
        vec![Arc::new(StringArray::from_iter_values(vec!["Tom"]))],
    )?;
    table.insert(TableInsertion::new(vec![batch])).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
| Tom  | 24  |
+------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn insert_unordered_schema(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let batch = RecordBatch::try_new(
        Arc::new(table.schema.project(&[2, 1])?),
        vec![
            Arc::new(Int32Array::from_iter_values(vec![24])),
            Arc::new(StringArray::from_iter_values(vec!["Tom"])),
        ],
    )?;
    table.insert(TableInsertion::new(vec![batch])).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
| Bob     | 21  |
| Charlie | 22  |
| David   | 23  |
| Tom     | 24  |
+---------+-----+"#,
    );

    Ok(())
}
