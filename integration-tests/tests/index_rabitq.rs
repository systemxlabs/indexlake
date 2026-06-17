use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::TableSearch;
use indexlake_integration_tests::data::prepare_simple_fixed_size_vector_table;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

use indexlake::table::IndexCreation;
use indexlake_index_rabitq::{RabitqIndexKind, RabitqIndexParams, RabitqMetric, RabitqSearchQuery};
use indexlake_integration_tests::utils::table_search;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn create_rabitq_index(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(RabitqIndexKind));

    let table = prepare_simple_fixed_size_vector_table(&client, format).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_creation = IndexCreation {
        name: "rabitq_index".to_string(),
        kind: RabitqIndexKind.kind().to_string(),
        key_columns: vec!["vector".to_string()],
        params: Arc::new(RabitqIndexParams {
            metric: RabitqMetric::L2,
            total_bits: 7,
        }),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(index_creation.clone()).await?;

    let search = TableSearch {
        query: Arc::new(RabitqSearchQuery {
            vector: vec![26.0, 26.0, 26.0],
        }),
        projection: None,
        dynamic_fields: vec!["score".to_string()],
        limit: Some(2),
        concurrency: 8,
    };

    let table = client.load_table(&namespace_name, &table_name).await?;
    let table_str = table_search(&table, search).await?;
    println!("{}", table_str);
    assert!(
        table_str.contains("30.0, 30.0, 30.0") || table_str.contains("20.0, 20.0, 20.0"),
        "expected search result to contain nearby vectors, got:\n{table_str}"
    );

    Ok(())
}
