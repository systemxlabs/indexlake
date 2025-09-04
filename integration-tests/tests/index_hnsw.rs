use arrow::array::{Float32Builder, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{TableConfig, TableCreation, TableInsertion, TableSearch};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int32Array, RecordBatch};
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

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", DataType::List(list_inner_field.clone()), true),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
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

    let table = client.load_table(&namespace_name, &table_name).await?;

    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[10.0, 10.0, 10.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[20.0, 20.0, 20.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[30.0, 30.0, 30.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[40.0, 40.0, 40.0]);
    list_builder.append(true);
    let list_array = list_builder.finish();

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(list_array),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let search = TableSearch {
        query: Arc::new(HnswSearchQuery {
            vector: vec![26.0, 26.0, 26.0],
            limit: 2,
        }),
        projection: None,
    };

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
