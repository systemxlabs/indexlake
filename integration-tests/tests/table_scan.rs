use arrow::{
    array::{Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use indexlake::expr::{col, lit};
use indexlake::table::{TableConfig, TableCreation, TableScan, TableScanPartition};
use indexlake::{
    Client,
    catalog::Catalog,
    storage::{DataFileFormat, Storage},
};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::table_scan;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::{collections::HashMap, sync::Arc};

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_projection(
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

    let scan = TableScan::default().with_projection(Some(vec![0, 2]));
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----+
| age |
+-----+
| 20  |
| 21  |
| 22  |
| 23  |
+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_filters(
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

    let scan = TableScan::default()
        .with_filters(vec![col("age").gt(lit(21)), col("name").eq(lit("Charlie"))]);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Charlie | 22  |
+---------+-----+"#
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn partitioned_scan(
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
    let table_config = TableConfig {
        inline_row_count_limit: 2,
        parquet_row_group_size: 1,
        preferred_data_file_format: format,
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

    // produce a data file
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(Int32Array::from(vec![20, 21])),
        ],
    )?;
    table.insert(&[record_batch]).await?;

    // produce a data file
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Charlie", "David"])),
            Arc::new(Int32Array::from(vec![22, 23])),
        ],
    )?;
    table.insert(&[record_batch]).await?;

    // insert a inline row
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Eva"])),
            Arc::new(Int32Array::from(vec![24])),
        ],
    )?;
    table.insert(&[record_batch]).await?;

    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let scan = TableScan::default().with_partition(TableScanPartition {
        partition_idx: 0,
        partition_count: 2,
    });
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| Alice | 20  |
| Bob   | 21  |
| Eva   | 24  |
+-------+-----+"#,
    );

    let scan = TableScan::default().with_partition(TableScanPartition {
        partition_idx: 1,
        partition_count: 2,
    });
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Charlie | 22  |
| David   | 23  |
+---------+-----+"#,
    );

    let scan = TableScan::default().with_partition(TableScanPartition {
        partition_idx: 2,
        partition_count: 3,
    });
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
+------+-----+"#,
    );

    Ok(())
}
