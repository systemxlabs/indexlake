use arrow::array::{RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::pretty::pretty_format_batches;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{ExecutionPlan, collect};
use datafusion::prelude::SessionContext;
use datafusion_proto::physical_plan::AsExecutionPlan;
use datafusion_proto::protobuf::PhysicalPlanNode;
use indexlake::Client;
use indexlake::catalog::{Catalog, INTERNAL_ROW_ID_FIELD_NAME, Scalar};
use indexlake::storage::{DataFileFormat, Storage};
use indexlake::table::{IndexCreation, TableConfig, TableCreation, TableInsertion, TableSearch};
use indexlake_datafusion::{
    IndexLakePhysicalCodec, IndexLakeSearchExec, IndexLakeTable, LazyTable,
};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::{
    datafusion_delete, datafusion_insert, datafusion_scan, datafusion_update,
    read_first_row_id_bytes_from_table, sort_record_batches,
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
async fn datafusion_full_scan(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table").await;
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
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
async fn test_datafusion_update(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    // Test update with filter
    let table_str = datafusion_update(
        &session,
        "UPDATE indexlake_table SET age = age + 1 WHERE name = 'Bob'",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 1     |
+-------+"#,
    );

    // Verify only Bob's age updated
    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table").await;
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
| Bob     | 22  |
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
async fn test_datafusion_delete(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    // Test delete with filter
    let table_str = datafusion_delete(&session, "DELETE FROM indexlake_table WHERE age > 21").await;
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 2     |
+-------+"#,
    );

    // Verify only rows with age > 21 deleted (Charlie:22, David:23)
    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table").await;
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| Alice | 20  |
| Bob   | 21  |
+-------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_with_projection(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let table_str = datafusion_scan(
        &session,
        "SELECT _indexlake_row_id, name FROM indexlake_table",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+---------+
| name    |
+---------+
| Alice   |
| Bob     |
| Charlie |
| David   |
+---------+"#,
    );

    let table_str = datafusion_scan(
        &session,
        "SELECT _indexlake_row_id, age, name FROM indexlake_table",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-----+---------+
| age | name    |
+-----+---------+
| 20  | Alice   |
| 21  | Bob     |
| 22  | Charlie |
| 23  | David   |
+-----+---------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_with_filters(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table where age > 21").await;
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
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
async fn datafusion_scan_hide_row_id_with_filters(
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

    let df_table =
        IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?.with_hide_row_id(true);
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let df = session
        .sql("SELECT * FROM indexlake_table where age > 22")
        .await?;
    let batches = df.collect().await?;
    let table_str = pretty_format_batches(&batches)?.to_string();
    println!("{}", table_str);

    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| David | 23  |
+-------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_with_row_id_filter(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let table_str = datafusion_scan(
        &session,
        &format!(
            "SELECT * FROM indexlake_table where _indexlake_row_id = X'{}'",
            hex::encode(first_row_id_bytes)
        ),
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-------+-----+
| name  | age |
+-------+-----+
| Alice | 20  |
+-------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_with_projection_filter(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let table_str = datafusion_scan(
        &session,
        "SELECT _indexlake_row_id, name FROM indexlake_table where age > 21",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+---------+
| name    |
+---------+
| Charlie |
| David   |
+---------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_with_limit(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let df = session.sql("SELECT * FROM indexlake_table limit 2").await?;
    let plan = df.create_physical_plan().await?;
    println!(
        "plan: {}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );
    let batches = collect(plan, session.task_ctx()).await?;
    let num_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
    assert_eq!(num_rows, 2);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_full_insert(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let table_str = datafusion_insert(
        &session,
        "INSERT INTO indexlake_table (name, age) VALUES ('Eve', 24)",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 1     |
+-------+"#,
    );

    let table_str = datafusion_insert(
        &session,
        "INSERT INTO indexlake_table (age, name) VALUES (25, 'Frank')",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 1     |
+-------+"#,
    );

    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table").await;
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
| Bob     | 21  |
| Charlie | 22  |
| David   | 23  |
| Eve     | 24  |
| Frank   | 25  |
+---------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_partial_insert(
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

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let default_values = HashMap::from([("age".to_string(), Scalar::from(24i32).into())]);
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        default_values,
        config: table_config,
        if_not_exists: false,
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let table_str = datafusion_insert(
        &session,
        "INSERT INTO indexlake_table (name) VALUES ('Eve')",
    )
    .await;
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 1     |
+-------+"#,
    );

    let table_str = datafusion_scan(&session, "SELECT * FROM indexlake_table").await;
    assert_eq!(
        table_str,
        r#"+------+-----+
| name | age |
+------+-----+
| Eve  | 24  |
+------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_scan_serialization(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Arc::new(Client::new(catalog, storage));
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(client.clone(), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let df = session.sql("SELECT * FROM indexlake_table").await?;
    let plan = df.create_physical_plan().await?;
    println!(
        "plan: {}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );

    let codec = IndexLakePhysicalCodec::new(client.clone());
    let mut plan_buf: Vec<u8> = vec![];
    let plan_proto = PhysicalPlanNode::try_from_physical_plan(plan, &codec)?;
    plan_proto.try_encode(&mut plan_buf)?;
    let new_plan: Arc<dyn ExecutionPlan> = PhysicalPlanNode::try_decode(&plan_buf)
        .and_then(|proto| proto.try_into_physical_plan(&session.task_ctx(), &codec))?;
    println!(
        "deserialized plan: {}",
        DisplayableExecutionPlan::new(new_plan.as_ref()).indent(true)
    );

    let batches = collect(new_plan, session.task_ctx()).await?;
    let mut sorted_batch = sort_record_batches(&batches, INTERNAL_ROW_ID_FIELD_NAME)?;
    sorted_batch.remove_column(0);
    let table_str = pretty_format_batches(&[sorted_batch])?.to_string();
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
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
async fn datafusion_insert_serialization(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Arc::new(Client::new(catalog, storage));
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(client.clone(), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let df = session
        .sql("INSERT INTO indexlake_table (name, age) VALUES ('Eve', 24)")
        .await?;
    let plan = df.create_physical_plan().await?;
    println!(
        "plan: {}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );

    let codec = IndexLakePhysicalCodec::new(client.clone());
    let mut plan_buf: Vec<u8> = vec![];
    let plan_proto = PhysicalPlanNode::try_from_physical_plan(plan, &codec)?;
    plan_proto.try_encode(&mut plan_buf)?;
    let new_plan: Arc<dyn ExecutionPlan> = PhysicalPlanNode::try_decode(&plan_buf)
        .and_then(|proto| proto.try_into_physical_plan(&session.task_ctx(), &codec))?;
    println!(
        "deserialized plan: {}",
        DisplayableExecutionPlan::new(new_plan.as_ref()).indent(true)
    );

    let batches = collect(new_plan, session.task_ctx()).await?;
    let table_str = pretty_format_batches(&batches)?.to_string();
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------+
| count |
+-------+
| 1     |
+-------+"#,
    );

    let df = session.sql("SELECT * FROM indexlake_table").await?;
    let batches = df.collect().await?;
    let mut sorted_batch = sort_record_batches(&batches, INTERNAL_ROW_ID_FIELD_NAME)?;
    sorted_batch.remove_column(0);
    let table_str = pretty_format_batches(&[sorted_batch])?.to_string();
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| Alice   | 20  |
| Bob     | 21  |
| Charlie | 22  |
| David   | 23  |
| Eve     | 24  |
+---------+-----+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_count1_with_filter(
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

    let df_table = IndexLakeTable::try_new(Arc::new(client), Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;

    let df = session
        .sql("SELECT COUNT(1) FROM indexlake_table where age > 21")
        .await?;
    let batches = df.collect().await?;
    let table_str = pretty_format_batches(&batches)?.to_string();
    println!("{table_str}");

    assert_eq!(
        table_str,
        r#"+-----------------+
| count(Int64(1)) |
+-----------------+
| 2               |
+-----------------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_search_exec(
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
    client.register_index_kind(Arc::new(indexlake_index_bm25::BM25IndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
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

    // Insert test data
    let table = client.load_table(&namespace_name, &table_name).await?;
    let batches = vec![RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                "Rust Guide",
                "Python Guide",
                "Rust vs Python",
            ])),
            Arc::new(StringArray::from(vec![
                "Rust is a systems programming language",
                "Python is a scripting language",
                "Comparing Rust and Python for data engineering",
            ])),
        ],
    )?];
    table.insert(TableInsertion::new(batches)).await?;

    // Create BM25 index (create_index takes ownership, so reload)
    let table = client.load_table(&namespace_name, &table_name).await?;
    let index_creation = IndexCreation {
        name: "content_idx".to_string(),
        kind: "bm25".to_string(),
        key_columns: vec!["content".to_string()],
        params: Arc::new(indexlake_index_bm25::BM25IndexParams { avgdl: 256. }),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(index_creation).await?;

    // Use BM25 search to find "Rust" documents
    let query = Arc::new(indexlake_index_bm25::BM25SearchQuery {
        query: "Rust".to_string(),
        limit: Some(2),
    });
    let dynamic_fields = vec!["score".to_string()];

    // Compute the correct exec schema via TableSearch::output_schema()
    let table = client.load_table(&namespace_name, &table_name).await?;
    let table_search = TableSearch {
        query: query.clone(),
        projection: None,
        dynamic_fields: dynamic_fields.clone(),
        concurrency: 8,
    };
    let exec_schema = table_search.output_schema(&table)?;

    let lazy_table =
        LazyTable::new(Arc::new(client), namespace_name, table_name).with_table(Arc::new(table));
    let exec =
        IndexLakeSearchExec::try_new(lazy_table, exec_schema.clone(), query, dynamic_fields, None)?;

    // Verify exec schema includes dynamic field
    let schema = exec.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
        field_names,
        vec!["_indexlake_row_id", "title", "content", "score"]
    );

    // Execute and collect results
    let session = SessionContext::new();
    let batches = collect(Arc::new(exec), session.task_ctx()).await?;
    let table_str = pretty_format_batches(&batches)?.to_string();
    println!("{table_str}");

    // Should have at most 2 results (limit=2), with score column
    assert!(batches.iter().map(|b| b.num_rows()).sum::<usize>() <= 2);
    assert!(batches[0].schema().index_of("score").is_ok());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn datafusion_search_exec_serialization(
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
    client.register_index_kind(Arc::new(indexlake_index_bm25::BM25IndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![Field::new(
        "title",
        DataType::Utf8,
        false,
    )]));
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
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

    // Insert test data
    let table = client.load_table(&namespace_name, &table_name).await?;
    let batches = vec![RecordBatch::try_new(
        table_schema.clone(),
        vec![Arc::new(StringArray::from(vec![
            "Rust Guide",
            "Python Guide",
        ]))],
    )?];
    table.insert(TableInsertion::new(batches)).await?;

    // Create BM25 index
    let table = client.load_table(&namespace_name, &table_name).await?;
    let index_creation = IndexCreation {
        name: "title_idx".to_string(),
        kind: "bm25".to_string(),
        key_columns: vec!["title".to_string()],
        params: Arc::new(indexlake_index_bm25::BM25IndexParams { avgdl: 256. }),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(index_creation).await?;

    let query = Arc::new(indexlake_index_bm25::BM25SearchQuery {
        query: "Rust".to_string(),
        limit: Some(1),
    });
    let dynamic_fields = vec!["score".to_string()];

    let table = client.load_table(&namespace_name, &table_name).await?;
    let table_search = TableSearch {
        query: query.clone(),
        projection: None,
        dynamic_fields: dynamic_fields.clone(),
        concurrency: 8,
    };
    let exec_schema = table_search.output_schema(&table)?;

    let client = Arc::new(client);
    let lazy_table = LazyTable::new(client.clone(), namespace_name.clone(), table_name.clone())
        .with_table(Arc::new(table));
    let exec = IndexLakeSearchExec::try_new(lazy_table, exec_schema, query, dynamic_fields, None)?;
    let exec = Arc::new(exec);

    // Serialize and deserialize
    let codec = IndexLakePhysicalCodec::new(client.clone());
    let mut plan_buf: Vec<u8> = vec![];
    let plan_proto = PhysicalPlanNode::try_from_physical_plan(exec.clone(), &codec)?;
    plan_proto.try_encode(&mut plan_buf)?;
    let new_plan: Arc<dyn ExecutionPlan> =
        PhysicalPlanNode::try_decode(&plan_buf).and_then(|proto| {
            proto.try_into_physical_plan(&SessionContext::new().task_ctx(), &codec)
        })?;

    // Verify both can execute
    let session = SessionContext::new();
    let batches = collect(exec, session.task_ctx()).await?;
    let deser_batches = collect(new_plan, session.task_ctx()).await?;

    assert_eq!(batches.len(), deser_batches.len());
    for (b1, b2) in batches.iter().zip(deser_batches.iter()) {
        assert_eq!(b1.num_rows(), b2.num_rows());
        assert_eq!(b1.num_columns(), b2.num_columns());
    }

    Ok(())
}
