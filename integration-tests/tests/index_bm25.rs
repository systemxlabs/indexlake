use arrow::{
    array::{BinaryArray, StringArray},
    datatypes::{DataType, Field, Schema},
};
use indexlake::{
    Client,
    catalog::Catalog,
    index::IndexKind,
    storage::Storage,
    table::{TableConfig, TableCreation, TableSearch},
};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

use arrow::array::RecordBatch;
use indexlake::table::IndexCreation;
use indexlake_index_bm25::{BM25IndexKind, BM25IndexParams, BM25SearchQuery};
use indexlake_integration_tests::utils::table_search;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn create_bm25_index(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BM25IndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("title", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        ..Default::default()
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
    };
    client.create_table(table_creation).await?;
    let mut table = client.load_table(&namespace_name, &table_name).await?;

    let index_creation = IndexCreation {
        name: "bm25_index".to_string(),
        kind: BM25IndexKind.kind().to_string(),
        key_columns: vec!["content".to_string()],
        params: Arc::new(BM25IndexParams { avgdl: 256. }),
    };
    table.create_index(index_creation.clone()).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                "title1", "title2", "title3", "title4", "title5", "title6", "title7",
            ])),
            Arc::new(StringArray::from(vec![
                "The sky blushed pink as the sun dipped below the horizon.",
                "She found a forgotten letter tucked inside an old book.",
                "Apples, oranges, pink grapefruits, and more pink grapefruits.",
                "A single drop of rain fell, followed by a thousand more.",
                "小明硕士毕业于中国科学院计算所，后在日本京都大学深造。",
                "张华考上了北京大学；李萍进了中国人民大学；我在百货公司当售货员：我们都有光明的前途。",
                "今天天气真不错，我去了公园，看到了很多花，很漂亮。",
            ])),
        ],
    )?;
    table.insert(&[record_batch]).await?;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let search = TableSearch {
        query: Arc::new(BM25SearchQuery {
            query: "pink".to_string(),
            limit: Some(2),
        }),
        projection: None,
    };
    let table_str = table_search(&table, search).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------------------+--------+---------------------------------------------------------------+
| _indexlake_row_id | title  | content                                                       |
+-------------------+--------+---------------------------------------------------------------+
| 3                 | title3 | Apples, oranges, pink grapefruits, and more pink grapefruits. |
| 1                 | title1 | The sky blushed pink as the sun dipped below the horizon.     |
+-------------------+--------+---------------------------------------------------------------+"#,
    );

    let search = TableSearch {
        query: Arc::new(BM25SearchQuery {
            query: "大学".to_string(),
            limit: Some(2),
        }),
        projection: None,
    };
    let table_str = table_search(&table, search).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------------------+--------+--------------------------------------------------------------------------------------+
| _indexlake_row_id | title  | content                                                                              |
+-------------------+--------+--------------------------------------------------------------------------------------+
| 6                 | title6 | 张华考上了北京大学；李萍进了中国人民大学；我在百货公司当售货员：我们都有光明的前途。 |
| 5                 | title5 | 小明硕士毕业于中国科学院计算所，后在日本京都大学深造。                               |
+-------------------+--------+--------------------------------------------------------------------------------------+"#,
    );

    Ok(())
}
