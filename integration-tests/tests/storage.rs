use indexlake::storage::Storage;
use indexlake_integration_tests::{storage_fs, storage_s3};
use std::sync::Arc;

#[rstest::rstest]
#[case(async { storage_fs() })]
#[case(async { storage_s3().await })]
#[tokio::test(flavor = "multi_thread")]
async fn file_operations(
    #[future(awt)]
    #[case]
    storage: Arc<dyn Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "test/test.txt";
    if storage.exists(file_path).await? {
        storage.delete(file_path).await?;
    }

    let mut output_file = storage.create(file_path).await?;
    let expected = bytes::Bytes::from("Hello, world!");
    output_file.write(expected.clone()).await?;
    output_file.close().await?;

    let input_file = storage.open(file_path).await?;
    let file_meta = input_file.metadata().await?;
    let bytes = input_file.read(0..file_meta.size).await?;
    assert_eq!(bytes, expected);

    storage.delete(file_path).await?;
    assert!(!storage.exists(file_path).await?);

    Ok(())
}
