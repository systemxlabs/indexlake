use arrow::array::*;
use arrow::datatypes::*;
use futures::TryStreamExt;
use indexlake::catalog::INTERNAL_ROW_ID_FIELD_REF;
use indexlake::index::IndexKind;
use indexlake::table::IndexCreation;
use indexlake::table::TableScan;
use indexlake::{
    Client,
    catalog::Catalog,
    storage::{DataFileFormat, Storage},
    table::{TableConfig, TableCreation},
};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};
use indexlake_integration_tests::data::prepare_simple_testing_table;
use indexlake_integration_tests::utils::full_table_scan;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::i128;
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
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    let expected_namespace_id = client.create_namespace(&namespace_name, true).await?;

    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: expected_schema.clone(),
        config: TableConfig::default(),
    };

    let expected_table_id = client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;
    println!("table: {:?}", table);
    assert_eq!(table.namespace_id, expected_namespace_id);
    assert_eq!(table.namespace_name, namespace_name);
    assert_eq!(table.table_id, expected_table_id);
    assert_eq!(table.table_name, table_name);

    let mut fields = vec![INTERNAL_ROW_ID_FIELD_REF.clone()];
    fields.extend(expected_schema.fields.iter().map(|f| f.clone()));
    let expected_schema = Schema::new(fields);

    assert_eq!(table.schema.as_ref(), &expected_schema);

    Ok(())
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
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("boolean_col", DataType::Boolean, true),
        Field::new("int8_col", DataType::Int8, true),
        Field::new("int16_col", DataType::Int16, true),
        Field::new("int32_col", DataType::Int32, true),
        Field::new("int64_col", DataType::Int64, true),
        Field::new("uint8_col", DataType::UInt8, true),
        Field::new("uint16_col", DataType::UInt16, true),
        Field::new("uint32_col", DataType::UInt32, true),
        Field::new("uint64_col", DataType::UInt64, true),
        Field::new("float32_col", DataType::Float32, true),
        Field::new("float64_col", DataType::Float64, true),
        Field::new(
            "timestamp_second_col",
            DataType::Timestamp(TimeUnit::Second, None),
            true,
        ),
        Field::new(
            "timestamp_millisecond_col",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        ),
        Field::new(
            "timestamp_microsecond_col",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            true,
        ),
        Field::new(
            "timestamp_nanosecond_col",
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            true,
        ),
        Field::new("date32_col", DataType::Date32, true),
        Field::new("date64_col", DataType::Date64, true),
        Field::new(
            "time32_second_col",
            DataType::Time32(TimeUnit::Second),
            true,
        ),
        Field::new(
            "time32_millisecond_col",
            DataType::Time32(TimeUnit::Millisecond),
            true,
        ),
        Field::new(
            "time64_microsecond_col",
            DataType::Time64(TimeUnit::Microsecond),
            true,
        ),
        Field::new(
            "time64_nanosecond_col",
            DataType::Time64(TimeUnit::Nanosecond),
            true,
        ),
        Field::new("binary_col", DataType::Binary, true),
        Field::new("fixed_size_binary_col", DataType::FixedSizeBinary(2), true),
        Field::new("large_binary_col", DataType::LargeBinary, true),
        Field::new("binary_view_col", DataType::BinaryView, true),
        Field::new("utf8_col", DataType::Utf8, true),
        Field::new("large_utf8_col", DataType::LargeUtf8, true),
        Field::new("utf8_view_col", DataType::Utf8View, true),
        Field::new(
            "list_int32_col",
            DataType::List(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Field::new(
            "fixed_size_list_int32_col",
            DataType::FixedSizeList(Arc::new(Field::new("item", DataType::Int32, true)), 2),
            true,
        ),
        Field::new(
            "large_list_int32_col",
            DataType::LargeList(Arc::new(Field::new("item", DataType::Int32, true))),
            true,
        ),
        Field::new("decimal128_col", DataType::Decimal128(38, 10), true),
        Field::new("decimal256_col", DataType::Decimal256(76, 10), true),
    ]));

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: TableConfig::default(),
    };

    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(BooleanArray::from(vec![Some(false), Some(true), None])),
            Arc::new(Int8Array::from(vec![Some(i8::MIN), Some(i8::MAX), None])),
            Arc::new(Int16Array::from(vec![Some(i16::MIN), Some(i16::MAX), None])),
            Arc::new(Int32Array::from(vec![Some(i32::MIN), Some(i32::MAX), None])),
            Arc::new(Int64Array::from(vec![Some(i64::MIN), Some(i64::MAX), None])),
            Arc::new(UInt8Array::from(vec![Some(u8::MIN), Some(u8::MAX), None])),
            Arc::new(UInt16Array::from(vec![
                Some(u16::MIN),
                Some(u16::MAX),
                None,
            ])),
            Arc::new(UInt32Array::from(vec![
                Some(u32::MIN),
                Some(u32::MAX),
                None,
            ])),
            Arc::new(UInt64Array::from(vec![
                Some(u64::MIN),
                Some(u64::MAX),
                None,
            ])),
            Arc::new(Float32Array::from(vec![
                Some(f32::MIN),
                Some(f32::MAX),
                None,
            ])),
            Arc::new(Float64Array::from(vec![
                Some(f64::MIN),
                Some(f64::MAX),
                None,
            ])),
            Arc::new(TimestampSecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(TimestampMillisecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(TimestampMicrosecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(TimestampNanosecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(Date32Array::from(vec![Some(0i32), Some(11111111i32), None])),
            Arc::new(Date64Array::from(vec![Some(0i64), Some(11111111i64), None])),
            Arc::new(Time32SecondArray::from(vec![
                Some(0i32),
                Some(1111i32),
                None,
            ])),
            Arc::new(Time32MillisecondArray::from(vec![
                Some(0i32),
                Some(11111111i32),
                None,
            ])),
            Arc::new(Time64MicrosecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(Time64NanosecondArray::from(vec![
                Some(0i64),
                Some(11111111i64),
                None,
            ])),
            Arc::new(BinaryArray::from_opt_vec(vec![
                Some(&vec![0u8, 1u8]),
                Some(&vec![0u8, 1u8]),
                None,
            ])),
            Arc::new(FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                vec![Some(vec![0u8, 1u8]), Some(vec![0u8, 1u8]), None].into_iter(),
                2,
            )?),
            Arc::new(LargeBinaryArray::from_opt_vec(vec![
                Some(&vec![0u8, 1u8]),
                Some(&vec![0u8, 1u8]),
                None,
            ])),
            Arc::new(BinaryViewArray::from_iter(vec![
                Some(&vec![0u8, 1u8]),
                Some(&vec![0u8, 1u8]),
                None,
            ])),
            Arc::new(StringArray::from(vec![Some("utf8"), Some("utf8"), None])),
            Arc::new(LargeStringArray::from(vec![
                Some("largeutf8"),
                Some("largeutf8"),
                None,
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("utf8view"),
                Some("utf8view"),
                None,
            ])),
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                Some(vec![Some(0i32), Some(1i32)]),
                Some(vec![Some(2i32), Some(3i32)]),
                None,
            ])),
            Arc::new(FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
                vec![
                    Some(vec![Some(0i32), Some(1i32)]),
                    Some(vec![Some(2i32), Some(3i32)]),
                    None,
                ]
                .into_iter(),
                2,
            )),
            Arc::new(LargeListArray::from_iter_primitive::<Int32Type, _, _>(
                vec![
                    Some(vec![Some(0i32), Some(1i32)]),
                    Some(vec![Some(2i32), Some(3i32)]),
                    None,
                ],
            )),
            Arc::new(Decimal128Array::from(vec![
                Some(i128::MIN),
                Some(i128::MAX),
                None,
            ])),
            Arc::new(Decimal256Array::from(vec![
                Some(i256::MIN),
                Some(i256::MAX),
                None,
            ])),
        ],
    )?;
    table.insert(&[record_batch]).await?;

    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------------------+-------------+----------+-----------+-------------+----------------------+-----------+------------+------------+----------------------+---------------+-------------------------+----------------------+---------------------------+----------------------------+-------------------------------+--------------+-------------------------+-------------------+------------------------+------------------------+-----------------------+------------+-----------------------+------------------+-----------------+----------+----------------+---------------+----------------+---------------------------+----------------------+------------------------------------------+--------------------------------------------------------------------------------+
| _indexlake_row_id | boolean_col | int8_col | int16_col | int32_col   | int64_col            | uint8_col | uint16_col | uint32_col | uint64_col           | float32_col   | float64_col             | timestamp_second_col | timestamp_millisecond_col | timestamp_microsecond_col  | timestamp_nanosecond_col      | date32_col   | date64_col              | time32_second_col | time32_millisecond_col | time64_microsecond_col | time64_nanosecond_col | binary_col | fixed_size_binary_col | large_binary_col | binary_view_col | utf8_col | large_utf8_col | utf8_view_col | list_int32_col | fixed_size_list_int32_col | large_list_int32_col | decimal128_col                           | decimal256_col                                                                 |
+-------------------+-------------+----------+-----------+-------------+----------------------+-----------+------------+------------+----------------------+---------------+-------------------------+----------------------+---------------------------+----------------------------+-------------------------------+--------------+-------------------------+-------------------+------------------------+------------------------+-----------------------+------------+-----------------------+------------------+-----------------+----------+----------------+---------------+----------------+---------------------------+----------------------+------------------------------------------+--------------------------------------------------------------------------------+
| 1                 | false       | -128     | -32768    | -2147483648 | -9223372036854775808 | 0         | 0          | 0          | 0                    | -3.4028235e38 | -1.7976931348623157e308 | 1970-01-01T00:00:00  | 1970-01-01T00:00:00       | 1970-01-01T00:00:00        | 1970-01-01T00:00:00           | 1970-01-01   | 1970-01-01T00:00:00     | 00:00:00          | 00:00:00               | 00:00:00               | 00:00:00              | 0001       | 0001                  | 0001             | 0001            | utf8     | largeutf8      | utf8view      | [0, 1]         | [0, 1]                    | [0, 1]               | -1701411834604692317316873037.1588410572 | -578960446186580977117854925043439539266349923328202820197287920039.5656481996 |
| 2                 | true        | 127      | 32767     | 2147483647  | 9223372036854775807  | 255       | 65535      | 4294967295 | 18446744073709551615 | 3.4028235e38  | 1.7976931348623157e308  | 1970-05-09T14:25:11  | 1970-01-01T03:05:11.111   | 1970-01-01T00:00:11.111111 | 1970-01-01T00:00:00.011111111 | +32391-03-11 | 1970-01-01T03:05:11.111 | 00:18:31          | 03:05:11.111           | 00:00:11.111111        | 00:00:00.011111111    | 0001       | 0001                  | 0001             | 0001            | utf8     | largeutf8      | utf8view      | [2, 3]         | [2, 3]                    | [2, 3]               | 1701411834604692317316873037.1588410572  | 578960446186580977117854925043439539266349923328202820197287920039.5656481996  |
| 3                 |             |          |           |             |                      |           |            |            |                      |               |                         |                      |                           |                            |                               |              |                         |                   |                        |                        |                       |            |                       |                  |                 |          |                |               |                |                           |                      |                                          |                                                                                |
+-------------------+-------------+----------+-----------+-------------+----------------------+-----------+------------+------------+----------------------+---------------+-------------------------+----------------------+---------------------------+----------------------------+-------------------------------+--------------+-------------------------+-------------------+------------------------+------------------------+-----------------------+------------+-----------------------+------------------+-----------------+----------+----------------+---------------+----------------+---------------------------+----------------------+------------------------------------------+--------------------------------------------------------------------------------+"#,
    );

    Ok(())
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
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let expected_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: expected_schema.clone(),
        config: TableConfig::default(),
    };

    client.create_table(table_creation.clone()).await?;
    let result = client.create_table(table_creation).await;
    assert!(result.is_err());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn truncate_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);
    let table = prepare_simple_testing_table(&client, DataFileFormat::ParquetV2).await?;

    table.truncate().await?;

    let scan = TableScan::default();
    let stream = table.scan(scan).await?;
    let batches = stream.try_collect::<Vec<_>>().await?;
    assert_eq!(batches.len(), 0);

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn drop_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let client = Client::new(catalog, storage);

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: TableConfig::default(),
    };
    client.create_table(table_creation.clone()).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )?;
    table.insert(&[record_batch.clone()]).await?;
    // avoid dump task failure due to inline row table dropped
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;

    table.drop().await?;
    assert!(
        client
            .load_table(&namespace_name, &table_name)
            .await
            .is_err()
    );

    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;
    table.insert(&[record_batch]).await?;
    let table_str = full_table_scan(&table).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+-------------------+----+-------+
| _indexlake_row_id | id | name  |
+-------------------+----+-------+
| 1                 | 1  | Alice |
| 2                 | 2  | Bob   |
+-------------------+----+-------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn duplicated_index_name(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: TableConfig::default(),
    };
    client.create_table(table_creation).await?;

    let mut table = client.load_table(&namespace_name, &table_name).await?;

    let index_creation = IndexCreation {
        name: uuid::Uuid::new_v4().to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["name".to_string()],
        params: Arc::new(BTreeIndexParams),
    };

    table.create_index(index_creation.clone()).await?;

    let result = table.create_index(index_creation).await;
    assert!(result.is_err());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn unsupported_index_kind(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let mut table = prepare_simple_testing_table(&client, DataFileFormat::ParquetV2).await?;

    let index_creation = IndexCreation {
        name: uuid::Uuid::new_v4().to_string(),
        kind: "unsupported_index_kind".to_string(),
        key_columns: vec!["name".to_string()],
        params: Arc::new(BTreeIndexParams),
    };

    let result = table.create_index(index_creation).await;
    assert!(result.is_err());

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, storage_fs())]
#[case(async { catalog_postgres().await }, storage_s3())]
#[tokio::test(flavor = "multi_thread")]
async fn drop_index(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[case] storage: Arc<Storage>,
) -> Result<(), Box<dyn std::error::Error>> {
    init_env_logger();

    let mut client = Client::new(catalog, storage);
    client.register_index_kind(Arc::new(BTreeIndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: TableConfig::default(),
    };
    client.create_table(table_creation).await?;

    let mut table = client.load_table(&namespace_name, &table_name).await?;

    let index_name = uuid::Uuid::new_v4().to_string();
    let index_creation = IndexCreation {
        name: index_name.to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["name".to_string()],
        params: Arc::new(BTreeIndexParams),
    };

    table.create_index(index_creation.clone()).await?;

    table.drop_index(&index_name, false).await?;

    let result = table.drop_index(&index_name, false).await;
    assert!(result.is_err());

    table.create_index(index_creation).await?;

    Ok(())
}
