use arrow::array::{Array, Float32Builder, Int32Array, ListBuilder, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use futures::TryStreamExt;
use indexlake::Client;
use indexlake::catalog::{Catalog, Scalar};
use indexlake::expr::{col, lit};
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{TableConfig, TableCreation, TableInsertion, TableScan, TableScanPartition};
use indexlake_integration_tests::data::{
    prepare_simple_testing_table, prepare_simple_testing_table2, prepare_simple_vector_table,
};
use indexlake_integration_tests::utils::{
    assert_data_file_count, assert_inline_row_count, table_scan,
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::collections::HashMap;
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_projection(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
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
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_filters(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
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
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_limit_offset(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table2(&client, format).await?;

    let scan = TableScan::default().with_offset(0).with_limit(1);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| Alice | 20  |
+-------+-----+"#
    );

    let scan = TableScan::default().with_offset(1).with_limit(1);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
| Bob  | 21  |
+------+-----+"#
    );

    let scan = TableScan::default().with_offset(2).with_limit(1);
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

    let scan = TableScan::default().with_offset(3).with_limit(1);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| David | 23  |
+-------+-----+"#
    );

    let scan = TableScan::default().with_offset(4).with_limit(1);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
| Eve  | 24  |
+------+-----+"#
    );
    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_limit_offset_filter(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let scan = TableScan::default()
        .with_filters(vec![col("age").gt(lit(20))])
        .with_limit(1)
        .with_offset(1);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
| Bob  | 21  |
+------+-----+"#
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_limit_offset_filter_across_data_files(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table2(&client, format).await?;

    // `age > 20` matches Bob(21), Charlie(22), David(23), Eve(24), Frank(25),
    // George(26), Hannah(27), Ivy(28). Bob and Charlie are inline rows; the
    // rest live in two data files of three rows each.
    let filters = vec![col("age").gt(lit(20))];

    // offset == 0: stream directly and stop once the limit is reached.
    let scan = TableScan::default()
        .with_filters(filters.clone())
        .with_offset(0)
        .with_limit(2);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Bob     | 21  |
| Charlie | 22  |
+---------+-----+"#
    );

    // offset crosses the inline row / data file boundary.
    let scan = TableScan::default()
        .with_filters(filters.clone())
        .with_offset(1)
        .with_limit(2);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Charlie | 22  |
| David   | 23  |
+---------+-----+"#
    );

    // offset crosses the boundary between the two data files.
    let scan = TableScan::default()
        .with_filters(filters.clone())
        .with_offset(4)
        .with_limit(2);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+--------+-----+
| name   | age |
+--------+-----+
| Frank  | 25  |
| George | 26  |
+--------+-----+"#
    );

    // limit exceeds the number of rows left after applying offset.
    let scan = TableScan::default()
        .with_filters(filters.clone())
        .with_offset(6)
        .with_limit(5);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+--------+-----+
| name   | age |
+--------+-----+
| Hannah | 27  |
| Ivy    | 28  |
+--------+-----+"#
    );

    // offset skips every matching row: each file is provably inside the
    // discard region and is skipped without reading its data columns.
    let scan = TableScan::default()
        .with_filters(filters)
        .with_offset(8)
        .with_limit(1);
    let stream = table.scan(scan).await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(
        batches.iter().map(|batch| batch.num_rows()).sum::<usize>(),
        0
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn auto_partition_scan(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
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
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // produce a data file
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Charlie", "David"])),
            Arc::new(Int32Array::from(vec![22, 23])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // insert a inline row
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Eva"])),
            Arc::new(Int32Array::from(vec![24])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    assert_inline_row_count(&table, |count| count == 1).await?;
    assert_data_file_count(&table, |count| count == 2).await?;

    let scan = TableScan::default().with_partition(TableScanPartition::Auto {
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

    let scan = TableScan::default().with_partition(TableScanPartition::Auto {
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

    let scan = TableScan::default().with_partition(TableScanPartition::Auto {
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

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn scan_with_catalog_unsupported_filter(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_vector_table(&client, format).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[20.0, 20.0, 20.0]);
    list_builder.append(true);
    let list_array = list_builder.finish();
    let scalar = Scalar::List(Arc::new(list_array));

    let scan = TableScan::default().with_filters(vec![col("vector").gt(lit(scalar))]);
    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------------------+
| id | vector             |
+----+--------------------+
| 3  | [30.0, 30.0, 30.0] |
| 4  | [40.0, 40.0, 40.0] |
+----+--------------------+"#,
    );

    Ok(())
}

const WIDE_INLINE_ROWS: i32 = 5000;

/// Insert `row_count` rows of ~1KB payload and keep *all* of them inline.
///
/// The wide payload matters: a tokio-postgres response is delivered in batches,
/// so a small inline table fits in a single batch and hides the connection
/// lifecycle bug. ~300KB of rows spans several batches.
async fn prepare_wide_inline_table(
    client: &Client,
    row_count: i32,
) -> indexlake::ILResult<(String, String, indexlake::table::Table)> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: TableConfig {
            inline_row_count_limit: 100_000,
            parquet_row_group_size: 1024,
            preferred_data_file_format: DataFileFormat::ParquetV2,
        },
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let payload = "x".repeat(1024);
    let record_batch = RecordBatch::try_new(
        table_schema,
        vec![
            Arc::new(Int32Array::from((0..row_count).collect::<Vec<_>>())),
            Arc::new(StringArray::from(vec![payload; row_count as usize])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    Ok((namespace_name, table_name, table))
}

async fn collect_ids(
    table: &indexlake::table::Table,
    scan: TableScan,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let duration = std::time::Duration::from_secs(120);
    let batches = tokio::time::timeout(duration, async {
        let stream = table.scan(scan).await?;
        stream.try_collect::<Vec<_>>().await
    })
    .await
    .expect("scan timed out: the catalog connection was most likely not held by the row stream")?;

    let mut ids = Vec::new();
    for batch in &batches {
        let column = batch
            .column_by_name("id")
            .expect("id column should be present")
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("id column should be Int32");
        ids.extend((0..column.len()).map(|idx| column.value(idx)));
    }
    Ok(ids)
}

/// A full inline scan must work with a pool that only holds a single connection.
///
/// Regression test for a scan that opened the inline row stream first (keeping a
/// connection busy with an unread response) and then asked the catalog for the
/// partitioned data file records, which needs a second connection: the checkout
/// of that second connection hung in `is_valid` until `connection_timeout`.
#[tokio::test(flavor = "multi_thread")]
async fn scan_inline_rows_with_single_connection_pool() -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let setup_client = Client::new(catalog_postgres().await, storage_fs());
    let (namespace_name, table_name, table) =
        prepare_wide_inline_table(&setup_client, WIDE_INLINE_ROWS).await?;
    assert_inline_row_count(&table, |count| count == WIDE_INLINE_ROWS as usize).await?;
    assert_data_file_count(&table, |count| count == 0).await?;

    // Same database and storage, but a pool of exactly one connection.
    let scan_client = Client::new(
        indexlake_integration_tests::catalog_postgres_with_pool_size(1).await,
        storage_fs(),
    );
    let table = scan_client.load_table(&namespace_name, &table_name).await?;

    let ids = collect_ids(&table, TableScan::default().with_projection(Some(vec![1]))).await?;
    assert_eq!(ids, (0..WIDE_INLINE_ROWS).collect::<Vec<_>>());

    Ok(())
}

/// Paginated inline scans must push the window down and stay correct with a
/// single-connection pool.
#[tokio::test(flavor = "multi_thread")]
async fn paginated_scan_inline_rows_with_single_connection_pool()
-> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let setup_client = Client::new(catalog_postgres().await, storage_fs());
    let (namespace_name, table_name, table) =
        prepare_wide_inline_table(&setup_client, WIDE_INLINE_ROWS).await?;
    assert_inline_row_count(&table, |count| count == WIDE_INLINE_ROWS as usize).await?;

    let scan_client = Client::new(
        indexlake_integration_tests::catalog_postgres_with_pool_size(1).await,
        storage_fs(),
    );
    let table = scan_client.load_table(&namespace_name, &table_name).await?;

    let page = |offset: usize, limit: usize| {
        TableScan::default()
            .with_projection(Some(vec![1]))
            .with_offset(offset)
            .with_limit(limit)
    };

    assert_eq!(collect_ids(&table, page(0, 2)).await?, vec![0, 1]);
    assert_eq!(collect_ids(&table, page(1, 3)).await?, vec![1, 2, 3]);
    assert_eq!(collect_ids(&table, page(5, 4)).await?, vec![5, 6, 7, 8]);
    // A window that runs past the end of the inline rows must not error out.
    let tail = WIDE_INLINE_ROWS as usize - 2;
    assert_eq!(
        collect_ids(&table, page(tail, 10)).await?,
        vec![WIDE_INLINE_ROWS - 2, WIDE_INLINE_ROWS - 1]
    );
    // Offset beyond the last inline row yields nothing.
    let beyond = WIDE_INLINE_ROWS as usize + 100;
    assert!(collect_ids(&table, page(beyond, 10)).await?.is_empty());

    Ok(())
}
