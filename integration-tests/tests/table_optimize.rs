use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{Int32Array, RecordBatch, StringArray},
    datatypes::{DataType, Field, Schema},
};
use futures::TryStreamExt;
use indexlake::{
    Client,
    catalog::Catalog,
    expr::{col, lit},
    storage::{DataFileFormat, DirEntry, EntryMode, Storage},
    table::{TableConfig, TableCreation, TableInsertion, TableOptimization, TableUpdate},
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite,
    data::prepare_simple_testing_table,
    init_env_logger, storage_fs, storage_s3,
    utils::{assert_data_file_count, assert_inline_row_count, full_table_scan, timestamp_millis},
};

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn cleanup_orphan_files(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage.clone());
    let table = prepare_simple_testing_table(&client, DataFileFormat::ParquetV2).await?;
    let namespace_id = table.namespace_id;
    let table_id = table.table_id;

    let table_dir = format!("{}/{}", namespace_id, table_id);

    let file_entry_count_fn = |entries: &[DirEntry]| {
        entries
            .iter()
            .filter(|e| matches!(e.metadata.mode, EntryMode::File))
            .count()
    };

    let entries = storage
        .list(&table_dir)
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(file_entry_count_fn(&entries), 1);

    // create an orphan file
    let orphan_file_path = format!("{}/orphan_test_file", table_dir);
    let mut orphan_file = storage.create(&orphan_file_path).await?;
    orphan_file.write("content".as_bytes().into()).await?;
    orphan_file.close().await?;

    let entries = storage
        .list(&table_dir)
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(file_entry_count_fn(&entries), 2);

    // cleanup orphan files
    table
        .optimize(TableOptimization::CleanupOrphanFiles {
            last_modified_before: timestamp_millis(),
        })
        .await?;

    let entries = storage
        .list(&table_dir)
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    assert_eq!(file_entry_count_fn(&entries), 1);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() })]
#[case(async { catalog_postgres().await }, async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn merge_data_files(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage.clone());

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 2,
        parquet_row_group_size: 1,
        preferred_data_file_format: DataFileFormat::ParquetV2,
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

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(Int32Array::from(vec![20, 21])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Charlie", "David"])),
            Arc::new(Int32Array::from(vec![22, 23])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    assert_inline_row_count(&table, |count| count == 0).await?;
    assert_data_file_count(&table, |count| count == 2).await?;

    let update = TableUpdate {
        set_map: HashMap::from([("name".to_string(), lit("John"))]),
        condition: col("age").eq(lit(20)),
    };
    table.update(update).await?;

    table.delete(col("age").eq(lit(23))).await?;

    assert_inline_row_count(&table, |count| count == 1).await?;
    assert_data_file_count(&table, |count| count == 2).await?;

    let optimization = TableOptimization::MergeDataFiles {
        valid_row_threshold: 2,
    };
    table.optimize(optimization).await?;

    assert_inline_row_count(&table, |count| count == 1).await?;
    assert_data_file_count(&table, |count| count == 1).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{table_str}");
    assert_eq!(
        table_str,
        r#"+---------+-----+
| name    | age |
+---------+-----+
| John    | 20  |
| Bob     | 21  |
| Charlie | 22  |
+---------+-----+"#
    );

    Ok(())
}
