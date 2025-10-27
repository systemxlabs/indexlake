use std::sync::Arc;

use indexlake::{
    Client,
    catalog::Catalog,
    storage::{DataFileFormat, Storage},
    table::TableAlter,
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, data::prepare_simple_testing_table, init_env_logger,
    storage_fs, storage_s3, utils::full_table_scan,
};
use uuid::Uuid;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn alter_rename_table(
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
    let namespace_name = table.namespace_name.to_string();
    let table_name = table.table_name.to_string();

    let new_table_name = Uuid::new_v4().to_string();
    let alter = TableAlter::RenameTable {
        new_name: new_table_name.clone(),
    };
    table.alter(alter).await?;

    let res = client.load_table(&namespace_name, &table_name).await;
    assert!(res.is_err());

    let table = client.load_table(&namespace_name, &new_table_name).await?;
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
+---------+-----+"#
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn alter_rename_column(
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
    let namespace_name = table.namespace_name.to_string();
    let table_name = table.table_name.to_string();

    let alter = TableAlter::RenameColumn {
        old_name: "name".to_string(),
        new_name: "new_name".to_string(),
    };
    table.alter(alter).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;
    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");

    assert_eq!(
        table_str,
        r#"+----------+-----+
| new_name | age |
+----------+-----+
| Alice    | 20  |
| Bob      | 21  |
| Charlie  | 22  |
| David    | 23  |
+----------+-----+"#
    );

    Ok(())
}
