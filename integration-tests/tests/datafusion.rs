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
use indexlake::table::{TableConfig, TableCreation};
use indexlake_datafusion::{IndexLakePhysicalCodec, IndexLakeTable};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::{
    datafusion_insert, datafusion_scan, read_first_row_id_bytes_from_table, sort_record_batches,
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
async fn datafusion_scan_with_projection(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?.with_hide_row_id(true);
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let first_row_id_bytes = read_first_row_id_bytes_from_table(&table).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
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
    let default_values = HashMap::from([("age".to_string(), Scalar::from(24i32))]);
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

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
    let session = SessionContext::new();
    session.register_table("indexlake_table", Arc::new(df_table))?;
    let df = session.sql("SELECT * FROM indexlake_table").await?;
    let plan = df.create_physical_plan().await?;
    println!(
        "plan: {}",
        DisplayableExecutionPlan::new(plan.as_ref()).indent(true)
    );

    let codec = IndexLakePhysicalCodec::new(Arc::new(client));
    let mut plan_buf: Vec<u8> = vec![];
    let plan_proto = PhysicalPlanNode::try_from_physical_plan(plan, &codec)?;
    plan_proto.try_encode(&mut plan_buf)?;
    let new_plan: Arc<dyn ExecutionPlan> = PhysicalPlanNode::try_decode(&plan_buf)
        .and_then(|proto| proto.try_into_physical_plan(&session, &session.runtime_env(), &codec))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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

    let codec = IndexLakePhysicalCodec::new(Arc::new(client));
    let mut plan_buf: Vec<u8> = vec![];
    let plan_proto = PhysicalPlanNode::try_from_physical_plan(plan, &codec)?;
    plan_proto.try_encode(&mut plan_buf)?;
    let new_plan: Arc<dyn ExecutionPlan> = PhysicalPlanNode::try_decode(&plan_buf)
        .and_then(|proto| proto.try_into_physical_plan(&session, &session.runtime_env(), &codec))?;
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
    storage: Arc<Storage>,
    #[case] format: DataFileFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, format).await?;

    let df_table = IndexLakeTable::try_new(Arc::new(table))?;
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
