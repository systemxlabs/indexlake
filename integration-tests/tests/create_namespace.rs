use indexlake::{LakeClient, catalog::Catalog, storage::Storage};
use indexlake_integration_tests::{catalog_postgres, catalog_sqlite, storage_fs, storage_s3};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn create_namespace(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = LakeClient::new(catalog, storage);

    let namespace_name = "create_namespace";
    let expected_namespace_id = client.create_namespace(namespace_name, false).await?;

    let namespace_id = client.get_namespace_id(namespace_name).await?;
    assert_eq!(namespace_id, Some(expected_namespace_id));

    let namespace_id = client.get_namespace_id("not_exists").await?;
    assert_eq!(namespace_id, None);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn duplicated_namespace_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = LakeClient::new(catalog, storage);

    let namespace_name = "duplicated_namespace_name";
    client.create_namespace(namespace_name, false).await?;
    let result = client.create_namespace(namespace_name, false).await;
    assert!(result.is_err());

    Ok(())
}
