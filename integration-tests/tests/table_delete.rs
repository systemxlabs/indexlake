use indexlake::Client;
use indexlake::catalog::{Catalog, INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use indexlake::expr::{col, lit};
use indexlake::storage::{DataFileFormat, Storage};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::{full_table_scan, read_first_row_id_bytes_from_table};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[cfg_attr(feature = "lance-format", case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0))]
#[tokio::test(flavor = "multi_thread")]
async fn delete_table_by_condition(
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

    let condition = col("age").gt(lit(21i32));
    table.delete(&condition).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| Alice | 20  |
| Bob   | 21  |
+-------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[cfg_attr(feature = "lance-format", case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0))]
#[tokio::test(flavor = "multi_thread")]
async fn delete_table_by_row_id(
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

    let first_row_id_bytes = read_first_row_id_bytes_from_table(&table).await?;
    let condition = col(INTERNAL_ROW_ID_FIELD_NAME).eq(lit(Scalar::FixedSizeBinary(
        16,
        Some(first_row_id_bytes.to_vec()),
    )));
    table.delete(&condition).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Bob     | 21  |
| Charlie | 22  |
| David   | 23  |
+---------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[cfg_attr(feature = "lance-format", case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0))]
#[tokio::test(flavor = "multi_thread")]
async fn delete_table_by_constant_condition(
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

    let false_condition = lit(1i32).eq(lit(2i32));
    table.delete(&false_condition).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
| Bob     | 21  |
| Charlie | 22  |
| David   | 23  |
+---------+-----+"#,
    );

    let true_condition = lit(1i32).eq(lit(1i32));
    table.delete(&true_condition).await?;

    let table_str = full_table_scan(&table).await?;
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
