use futures::TryStreamExt;
use indexlake::record::{Scalar, pretty_print_rows};
use indexlake::{
    LakeClient,
    catalog::Catalog,
    record::{DataType, Field, Schema},
    storage::Storage,
    table::{TableConfig, TableCreation},
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn create_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) {
    init_env_logger();

    let client = LakeClient::new(catalog, storage);

    let namespace_name = "test_namespace";
    let expected_namespace_id = client.create_namespace(namespace_name).await.unwrap();

    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let table_name = "test_table";
    let table_creation = TableCreation {
        namespace_name: namespace_name.to_string(),
        table_name: table_name.to_string(),
        schema: expected_schema.clone(),
        config: TableConfig::default(),
    };

    let expected_table_id = client.create_table(table_creation).await.unwrap();

    let table = client.load_table(namespace_name, table_name).await.unwrap();
    println!("table: {:?}", table);
    assert_eq!(table.namespace_id, expected_namespace_id);
    assert_eq!(table.namespace_name, namespace_name);
    assert_eq!(table.table_id, expected_table_id);
    assert_eq!(table.table_name, table_name);
    let schema = table.schema.without_row_id();
    assert_eq!(
        schema
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>(),
        expected_schema
            .fields
            .iter()
            .map(|f| f.name.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        schema
            .fields
            .iter()
            .map(|f| f.data_type.clone())
            .collect::<Vec<_>>(),
        expected_schema
            .fields
            .iter()
            .map(|f| f.data_type.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        schema.fields.iter().map(|f| f.nullable).collect::<Vec<_>>(),
        expected_schema
            .fields
            .iter()
            .map(|f| f.nullable)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        schema
            .fields
            .iter()
            .map(|f| f.default_value.clone())
            .collect::<Vec<_>>(),
        expected_schema
            .fields
            .iter()
            .map(|f| f.default_value.clone())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        schema
            .fields
            .iter()
            .map(|f| f.metadata.clone())
            .collect::<Vec<_>>(),
        expected_schema
            .fields
            .iter()
            .map(|f| f.metadata.clone())
            .collect::<Vec<_>>()
    );
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn table_data_types(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) {
    init_env_logger();

    let client = LakeClient::new(catalog, storage);

    let namespace_name = "test_namespace";
    client.create_namespace(namespace_name).await.unwrap();

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("int16_col", DataType::Int16, true),
        Field::new("int32_col", DataType::Int32, true),
        Field::new("int64_col", DataType::Int64, true),
        Field::new("float32_col", DataType::Float32, true),
        Field::new("float64_col", DataType::Float64, true),
        Field::new("utf8_col", DataType::Utf8, true),
        Field::new("binary_col", DataType::Binary, true),
        Field::new("boolean_col", DataType::Boolean, true),
    ]));

    let table_name = "test_table";
    let table_creation = TableCreation {
        namespace_name: namespace_name.to_string(),
        table_name: table_name.to_string(),
        schema: table_schema.clone(),
        config: TableConfig::default(),
    };

    client.create_table(table_creation).await.unwrap();

    let table = client.load_table(namespace_name, table_name).await.unwrap();

    let columns = vec![
        "int16_col".to_string(),
        "int32_col".to_string(),
        "int64_col".to_string(),
        "float32_col".to_string(),
        "float64_col".to_string(),
        "utf8_col".to_string(),
        "binary_col".to_string(),
        "boolean_col".to_string(),
    ];
    let values = vec![vec![
        Scalar::Int16(Some(1)),
        Scalar::Int32(Some(1)),
        Scalar::Int64(Some(2)),
        Scalar::Float32(Some(1.1)),
        Scalar::Float64(Some(2.2)),
        Scalar::Utf8(Some("utf8".to_string())),
        Scalar::Binary(Some(vec![0, 1])),
        Scalar::Boolean(Some(true)),
    ]];
    table.insert(&columns, values).await.unwrap();

    let row_stream = table.scan().await.unwrap();
    let rows = row_stream.try_collect::<Vec<_>>().await.unwrap();
    let table_str = pretty_print_rows(Some(table_schema.clone()), &rows).to_string();
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----------+-----------+-----------+-------------+-------------+----------+------------+-------------+
| int16_col | int32_col | int64_col | float32_col | float64_col | utf8_col | binary_col | boolean_col |
+-----------+-----------+-----------+-------------+-------------+----------+------------+-------------+
| 1         | 1         | 2         | 1.1         | 2.2         | utf8     | 0001       | true        |
+-----------+-----------+-----------+-------------+-------------+----------+------------+-------------+"#,
    );
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn duplicated_table_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) {
    init_env_logger();

    let client = LakeClient::new(catalog, storage);

    let namespace_name = "test_namespace";
    client.create_namespace(namespace_name).await.unwrap();

    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let table_name = "test_table";
    let table_creation = TableCreation {
        namespace_name: namespace_name.to_string(),
        table_name: table_name.to_string(),
        schema: expected_schema.clone(),
        config: TableConfig::default(),
    };

    client.create_table(table_creation.clone()).await.unwrap();
    let result = client.create_table(table_creation).await;
    assert!(result.is_err());
}
