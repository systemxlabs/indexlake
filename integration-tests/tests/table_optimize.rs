use std::sync::Arc;

use futures::TryStreamExt;
use indexlake::{
    Client,
    catalog::Catalog,
    storage::{DataFileFormat, DirEntry, EntryMode, Storage},
    table::TableOptimization,
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, data::prepare_simple_testing_table, init_env_logger,
    storage_fs, storage_s3, utils::timestamp_millis,
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
    println!("LWZTEST entries: {entries:?}");
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
