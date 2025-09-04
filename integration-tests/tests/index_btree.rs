use std::sync::Arc;

use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::catalog::Scalar;
use indexlake::expr::{col, lit};
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{IndexCreation, TableScan};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};
use indexlake_integration_tests::data::{prepare_btree_integer_table, prepare_btree_string_table};
use indexlake_integration_tests::utils::table_scan;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn create_btree_index_integer(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let table = prepare_btree_integer_table(&client, format).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_creation = IndexCreation {
        name: "btree_index".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["integer".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        if_not_exists: false,
    };
    table.create_index(index_creation.clone()).await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let table = client.load_table(&namespace_name, &table_name).await?;

    // test point query
    let scan =
        TableScan::default().with_filters(vec![col("integer").eq(lit(Scalar::Int32(Some(75))))]);

    let table_str = table_scan(&table, scan).await?;
    println!("point query result:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+---------+
| id | integer |
+----+---------+
| 3  | 75      |
| 6  | 75      |
+----+---------+"#
    );

    // test range query
    let scan = TableScan::default().with_filters(vec![
        col("integer").gteq(lit(Scalar::Int32(Some(50)))),
        col("integer").lteq(lit(Scalar::Int32(Some(90)))),
    ]);

    let table_str = table_scan(&table, scan).await?;
    println!("range query result:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+---------+
| id | integer |
+----+---------+
| 2  | 50      |
| 3  | 75      |
| 5  | 90      |
| 6  | 75      |
+----+---------+"#
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn create_btree_index_string(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let table = prepare_btree_string_table(&client, format).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_creation = IndexCreation {
        name: "btree_index".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["string".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        if_not_exists: false,
    };
    table.create_index(index_creation.clone()).await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let table = client.load_table(&namespace_name, &table_name).await?;

    // test point query
    let scan = TableScan::default().with_filters(vec![
        col("string").eq(lit(Scalar::Utf8(Some("cherry".to_string())))),
    ]);

    let table_str = table_scan(&table, scan).await?;
    println!("string point query result:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+
| id | string |
+----+--------+
| 3  | cherry |
| 6  | cherry |
+----+--------+"#
    );

    // test range query
    let scan = TableScan::default().with_filters(vec![
        col("string").gteq(lit(Scalar::Utf8(Some("banana".to_owned())))),
        col("string").lteq(lit(Scalar::Utf8(Some("date".to_owned())))),
    ]);

    let table_str = table_scan(&table, scan).await?;
    println!("string range query result:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+
| id | string |
+----+--------+
| 2  | banana |
| 3  | cherry |
| 4  | date   |
| 6  | cherry |
+----+--------+"#
    );

    Ok(())
}
