use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::IndexCreation;
use indexlake_index_rstar::{RStarIndexKind, RStarIndexParams, WkbDialect};
use indexlake_integration_tests::data::prepare_simple_geom_table;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn duplicated_index_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RStarIndexKind));

    let table = prepare_simple_geom_table(&client, DataFileFormat::ParquetV2).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let mut index_creation = IndexCreation {
        name: uuid::Uuid::new_v4().to_string(),
        kind: RStarIndexKind.kind().to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        if_not_exists: false,
    };

    table.create_index(index_creation.clone()).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;
    let result = table.create_index(index_creation.clone()).await;
    assert!(result.is_err());

    index_creation.if_not_exists = true;
    let table = client.load_table(&namespace_name, &table_name).await?;
    table.create_index(index_creation).await?;

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn unsupported_index_kind(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RStarIndexKind));

    let table = prepare_simple_geom_table(&client, DataFileFormat::ParquetV2).await?;

    let index_creation = IndexCreation {
        name: uuid::Uuid::new_v4().to_string(),
        kind: "unsupported_index_kind".to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        if_not_exists: false,
    };

    let result = table.create_index(index_creation).await;
    assert!(result.is_err());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn drop_index(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RStarIndexKind));

    let table = prepare_simple_geom_table(&client, DataFileFormat::ParquetV2).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_name = uuid::Uuid::new_v4().to_string();
    let index_creation = IndexCreation {
        name: index_name.to_string(),
        kind: RStarIndexKind.kind().to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        if_not_exists: false,
    };

    table.create_index(index_creation.clone()).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;
    table.drop_index(&index_name, false).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;
    let result = table.drop_index(&index_name, false).await;
    assert!(result.is_err());

    let table = client.load_table(&namespace_name, &table_name).await?;
    table.create_index(index_creation).await?;

    Ok(())
}
