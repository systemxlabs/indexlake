use arrow::array::{Float32Builder, ListBuilder};
use arrow::datatypes::{DataType, Field};
use indexlake::Client;
use indexlake::catalog::{
    Catalog, CatalogDataType, CatalogSchema, Column, INTERNAL_ROW_ID_FIELD_NAME, Scalar,
};
use indexlake::expr::{col, lit};
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{TableScan, TableUpdate};
use indexlake_index_btree::BTreeIndexKind;
use indexlake_integration_tests::data::{
    prepare_simple_testing_table, prepare_simple_vector_table, prepare_table_with_two_btree_indexes,
};
use indexlake_integration_tests::utils::{
    full_table_scan, read_first_row_id_bytes_from_table, table_scan,
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::collections::HashMap;
use std::sync::Arc;
use uuid::Uuid;

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
+----+--------------------+"#,
    );
    Ok(())
}

async fn count_inline_index_records(
    catalog: Arc<dyn Catalog>,
    table_id: &Uuid,
) -> indexlake::ILResult<i64> {
    let schema = Arc::new(CatalogSchema::new(vec![Column::new(
        "index_id",
        CatalogDataType::Uuid,
        false,
    )]));
    let mut stream = catalog
        .query(
            &format!(
                "SELECT index_id FROM indexlake_index WHERE table_id = {}",
                catalog.sql_uuid_literal(table_id)
            ),
            schema,
        )
        .await?;
    let mut index_ids = Vec::new();
    while let Some(row) = futures::TryStreamExt::try_next(&mut stream).await? {
        index_ids.push(row.uuid(0)?.expect("index_id is not null"));
    }
    if index_ids.is_empty() {
        return Ok(0);
    }
    let index_id_literals: Vec<String> = index_ids
        .iter()
        .map(|id| catalog.sql_uuid_literal(id))
        .collect();
    let count_schema = Arc::new(CatalogSchema::new(vec![Column::new(
        "count",
        CatalogDataType::Int64,
        false,
    )]));
    let mut stream = catalog
        .query(
            &format!(
                "SELECT COUNT(1) FROM indexlake_inline_index WHERE index_id IN ({})",
                index_id_literals.join(", ")
            ),
            count_schema,
        )
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

    let table = prepare_table_with_two_btree_indexes(&client, format).await?;
    let catalog = client.catalog.clone();
    assert_eq!(
        count_inline_index_records(catalog.clone(), &table.table_id).await?,
        2
    );

    // Use a catalog-unsupported condition (List literal comparison) so the
    // inline update path deletes matched rows and reinserts updated batches.
    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[35.0, 35.0, 35.0]);
    list_builder.append(true);
    let scalar = Scalar::List(Arc::new(list_builder.finish()));

    // Update non-index column 'id' for the row with vector [40.0, 40.0, 40.0]
    let update = TableUpdate {
        set_map: HashMap::from([("id".to_string(), lit(100i32))]),
        condition: col("vector").gt(lit(scalar)),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);
    assert_eq!(
        count_inline_index_records(catalog, &table.table_id).await?,
        2
    );

    // Query via name index - should return exactly 1 row for David
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("David".to_string())))),
    ]);
    let table_str = table_scan(&table, scan).await?;
    println!("name index query after non-index update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----+-------+-----+--------------------+
| id  | name  | age | vector             |
+-----+-------+-----+--------------------+
| 100 | David | 23  | [40.0, 40.0, 40.0] |
+-----+-------+-----+--------------------+"#
    );

    // Query via age index - should return exactly 1 row for age=23
    let scan = TableScan::default().with_filters(vec![col("age").eq(lit(Scalar::Int32(Some(23))))]);
    let table_str = table_scan(&table, scan).await?;
    println!("age index query after non-index update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+-----+-------+-----+--------------------+
| id  | name  | age | vector             |
+-----+-------+-----+--------------------+
| 100 | David | 23  | [40.0, 40.0, 40.0] |
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

    let table = prepare_table_with_two_btree_indexes(&client, format).await?;
    let catalog = client.catalog.clone();
    assert_eq!(
        count_inline_index_records(catalog.clone(), &table.table_id).await?,
        2
    );

    // Use a catalog-unsupported condition (List literal comparison) so the
    // inline update path deletes matched rows and reinserts updated batches.
    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[35.0, 35.0, 35.0]);
    list_builder.append(true);
    let scalar = Scalar::List(Arc::new(list_builder.finish()));

    // Update indexed column 'name' only for the row with vector [40.0, 40.0, 40.0]
    let update = TableUpdate {
        set_map: HashMap::from([(
            "name".to_string(),
            lit(Scalar::Utf8(Some("XDavid".to_string()))),
        )]),
        condition: col("vector").gt(lit(scalar)),
    };
    let update_count = table.update(update).await?;
    assert_eq!(update_count, 1);
    assert_eq!(
        count_inline_index_records(catalog, &table.table_id).await?,
        3
    );

    // Query via name index for new name - should return exactly 1 row
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("XDavid".to_string())))),
    ]);
    let table_str = table_scan(&table, scan).await?;
    println!("name index query for new name after update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+-----+--------------------+
| id | name   | age | vector             |
+----+--------+-----+--------------------+
| 4  | XDavid | 23  | [40.0, 40.0, 40.0] |
+----+--------+-----+--------------------+"#
    );

    // Query via name index for old name - should return nothing
    let scan = TableScan::default().with_filters(vec![
        col("name").eq(lit(Scalar::Utf8(Some("David".to_string())))),
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

    // Query via age index - should still work correctly, returning 1 row for age=23
    let scan = TableScan::default().with_filters(vec![col("age").eq(lit(Scalar::Int32(Some(23))))]);
    let table_str = table_scan(&table, scan).await?;
    println!("age index query after name update:\n{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------+-----+--------------------+
| id | name   | age | vector             |
+----+--------+-----+--------------------+
| 4  | XDavid | 23  | [40.0, 40.0, 40.0] |
+----+--------+-----+--------------------+"#
    );

    Ok(())
}
