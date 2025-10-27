use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::storage::Storage;
use indexlake_integration_tests::{catalog_postgres, catalog_sqlite, storage_fs, storage_s3};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn create_namespace(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    let expected_namespace_id = client.create_namespace(&namespace_name, false).await?;

    let namespace_id = client.get_namespace_id(&namespace_name).await?;
    assert_eq!(namespace_id, Some(expected_namespace_id));

    let namespace_id = client.get_namespace_id("not_exists").await?;
    assert_eq!(namespace_id, None);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn duplicated_namespace_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, false).await?;
    let result = client.create_namespace(&namespace_name, false).await;
    assert!(result.is_err());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn rename_namespace_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, false).await?;

    let new_namespace_name = uuid::Uuid::new_v4().to_string();
    client
        .rename_namespace(&namespace_name, &new_namespace_name)
        .await?;

    let namespace_id = client.get_namespace_id(&namespace_name).await?;
    assert!(namespace_id.is_none());
    let namespace_id = client.get_namespace_id(&new_namespace_name).await?;
    assert!(namespace_id.is_some());

    Ok(())
}
