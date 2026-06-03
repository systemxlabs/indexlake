use arrow::array::{Float32Builder, Int32Array, ListBuilder, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use indexlake::Client;
use indexlake::catalog::{
    Catalog, CatalogDataType, CatalogSchema, Column, INTERNAL_ROW_ID_FIELD_NAME, Scalar,
};
use indexlake::expr::{col, lit};
use indexlake::index::IndexKind;
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{
    IndexCreation, TableConfig, TableCreation, TableInsertion, TableScan, TableUpdate,
};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};
use indexlake_integration_tests::data::{
    prepare_simple_testing_table, prepare_simple_vector_table,
};
use indexlake_integration_tests::utils::{
    full_table_scan, read_first_row_id_bytes_from_table, table_scan,
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::collections::HashMap;
use std::sync::Arc;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn update_table_by_condition(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let update = TableUpdate {
        set_map: HashMap::from([("age".to_string(), lit(30i32))]),
        condition: col("name").eq(lit("Alice")),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 30  |
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
#[tokio::test(flavor = "multi_thread")]
async fn update_table_by_row_id(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
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

    let update = TableUpdate {
        set_map: HashMap::from([("age".to_string(), lit(30i32))]),
        condition,
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 30  |
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
#[tokio::test(flavor = "multi_thread")]
async fn update_table_by_catalog_unsupported_condition(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_vector_table(&client, format).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[20.0, 20.0, 20.0]);
    list_builder.append(true);
    let list_array = list_builder.finish();
    let scalar = Scalar::List(Arc::new(list_array));

    let update = TableUpdate {
        set_map: HashMap::from([("id".to_string(), lit(10i32))]),
        condition: col("vector").gt(lit(scalar)),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 2);

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------------------+
| id | vector             |
+----+--------------------+
| 1  | [10.0, 10.0, 10.0] |
| 2  | [20.0, 20.0, 20.0] |
| 10 | [30.0, 30.0, 30.0] |
| 10 | [40.0, 40.0, 40.0] |
+----+--------------------+
"#,
    );
    Ok(())
}

async fn prepare_table_with_two_btree_indexes(
    client: &Client,
    format: DataFileFormat,
) -> Result<(indexlake::table::Table, Arc<dyn Catalog>), Box<dyn std::error::Error>> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
        Field::new_list(
            "vector",
            Arc::new(Field::new("item", DataType::Float32, false)),
            false,
        ),
    ]));
    // Keep rows inline so inline indexes are used
    let table_config = TableConfig {
        inline_row_count_limit: 100,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut vector_builder = ListBuilder::new(Float32Builder::new()).with_field(list_inner_field);
    for vector in [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [40.0, 40.0, 40.0],
    ] {
        vector_builder.values().append_slice(&vector);
        vector_builder.append(true);
    }

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "David"])),
            Arc::new(Int32Array::from(vec![20, 21, 22, 23])),
            Arc::new(vector_builder.finish()),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // Create btree index on name
    let name_index_creation = IndexCreation {
        name: "name_btree".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["name".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(name_index_creation).await?;

    // Reload table after create_index consumes self
    let table = client.load_table(&namespace_name, &table_name).await?;

    // Create btree index on age
    let age_index_creation = IndexCreation {
        name: "age_btree".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["age".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(age_index_creation).await?;

    // Reload table after create_index consumes self
    let table = client.load_table(&namespace_name, &table_name).await?;
    Ok((table, client.catalog.clone()))
}

async fn count_inline_index_records(catalog: Arc<dyn Catalog>) -> indexlake::ILResult<i64> {
    let schema = Arc::new(CatalogSchema::new(vec![Column::new(
        "count",
        CatalogDataType::Int64,
        false,
    )]));
    let mut stream = catalog
        .query("SELECT COUNT(1) FROM indexlake_inline_index", schema)
        .await?;
    let row = futures::TryStreamExt::try_next(&mut stream)
        .await?
        .expect("count query returns one row");
    Ok(row.int64(0)?.expect("count is not null"))
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn update_non_index_column_does_not_affect_indexes(
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
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let (table, catalog) = prepare_table_with_two_btree_indexes(&client, format).await?;
    assert_eq!(count_inline_index_records(catalog.clone()).await?, 2);

    // Use a catalog-unsupported condition so the inline update path deletes
    // matched rows and reinserts updated batches.
    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[5.0, 5.0, 5.0]);
    list_builder.append(true);
    let scalar = Scalar::List(Arc::new(list_builder.finish()));

    // Update non-index column 'id'
    let update = TableUpdate {
        set_map: HashMap::from([("id".to_string(), lit(100i32))]),
        condition: col("vector").gt(lit(scalar)).and(col("id").eq(lit(1i32))),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);
    assert_eq!(count_inline_index_records(catalog).await?, 2);

    // Query via name index - should return exactly 1 row for Alice
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("Alice".to_string())))),
    ]);
    let table_str = table_scan(&table, scan).await?;
    println!("name index query after non-index update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----+-------+-----+--------------------+
| id  | name  | age | vector             |
+-----+-------+-----+--------------------+
| 100 | Alice | 20  | [10.0, 10.0, 10.0] |
+-----+-------+-----+--------------------+"#
    );

    // Query via age index - should return exactly 1 row for age=20
    let scan = TableScan::default().with_filters(vec![col("age").eq(lit(Scalar::Int32(Some(20))))]);
    let table_str = table_scan(&table, scan).await?;
    println!("age index query after non-index update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----+-------+-----+--------------------+
| id  | name  | age | vector             |
+-----+-------+-----+--------------------+
| 100 | Alice | 20  | [10.0, 10.0, 10.0] |
+-----+-------+-----+--------------------+"#
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn update_one_index_column_only_affects_that_index(
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
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let (table, catalog) = prepare_table_with_two_btree_indexes(&client, format).await?;
    assert_eq!(count_inline_index_records(catalog.clone()).await?, 2);

    // Use a catalog-unsupported condition so the inline update path deletes
    // matched rows and reinserts updated batches.
    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[5.0, 5.0, 5.0]);
    list_builder.append(true);
    let scalar = Scalar::List(Arc::new(list_builder.finish()));

    // Update indexed column 'name' only
    let update = TableUpdate {
        set_map: HashMap::from([(
            "name".to_string(),
            lit(Scalar::Utf8(Some("Alicia".to_string()))),
        )]),
        condition: col("vector").gt(lit(scalar)).and(col("id").eq(lit(1i32))),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);
    assert_eq!(count_inline_index_records(catalog).await?, 2);

    // Query via name index for new name - should return exactly 1 row
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("Alicia".to_string())))),
    ]);
    let table_str = table_scan(&table, scan).await?;
    println!("name index query for new name after update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+-----+--------------------+
| id | name   | age | vector             |
+----+--------+-----+--------------------+
| 1  | Alicia | 20  | [10.0, 10.0, 10.0] |
+----+--------+-----+--------------------+"#
    );

    // Query via name index for old name - should return nothing
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("Alice".to_string())))),
    ]);
    let table_str = table_scan(&table, scan).await?;
    println!("name index query for old name after update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+------+-----+--------+
| id | name | age | vector |
+----+------+-----+--------+
+----+------+-----+--------+"#
    );

    // Query via age index - should still work correctly, returning 1 row for age=20
    let scan = TableScan::default().with_filters(vec![col("age").eq(lit(Scalar::Int32(Some(20))))]);
    let table_str = table_scan(&table, scan).await?;
    println!("age index query after name update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+-----+--------------------+
| id | name   | age | vector             |
+----+--------+-----+--------------------+
| 1  | Alicia | 20  | [10.0, 10.0, 10.0] |
+----+--------+-----+--------------------+"#
    );

    Ok(())
}
