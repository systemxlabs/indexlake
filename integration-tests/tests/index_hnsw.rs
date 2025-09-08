use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::TableSearch;
use indexlake_integration_tests::data::prepare_simple_vector_table;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

use indexlake::table::IndexCreation;
use indexlake_index_hnsw::{HnswIndexKind, HnswIndexParams, HnswSearchQuery};
use indexlake_integration_tests::utils::table_search;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[cfg_attr(feature = "lance-format", case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0))]
#[tokio::test(flavor = "multi_thread")]
async fn create_hnsw_index(
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
    client.register_index_kind(Arc::new(HnswIndexKind));

    let table = prepare_simple_vector_table(&client, format).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_creation = IndexCreation {
        name: "hnsw_index".to_string(),
        kind: HnswIndexKind.kind().to_string(),
        key_columns: vec!["vector".to_string()],
        params: Arc::new(HnswIndexParams {
            ef_construction: 400,
        }),
        if_not_exists: false,
    };
    table.create_index(index_creation.clone()).await?;

    let search = TableSearch {
        query: Arc::new(HnswSearchQuery {
            vector: vec![26.0, 26.0, 26.0],
            limit: 2,
        }),
        projection: None,
    };

    let table = client.load_table(&namespace_name, &table_name).await?;
    let table_str = table_search(&table, search).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------------------+
| id | vector             |
+----+--------------------+
| 3  | [30.0, 30.0, 30.0] |
| 2  | [20.0, 20.0, 20.0] |
+----+--------------------+"#,
    );

    Ok(())
}
