use arrow::array::{BinaryArray, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use futures::TryStreamExt;
use indexlake::Client;
use indexlake::catalog::Catalog;
use indexlake::index::IndexKind;
use indexlake::storage::Storage;
use indexlake::table::TableScan;
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

use geo::{Geometry, LineString, Point};
use geozero::{CoordDimensions, ToWkb};
use indexlake::expr::{col, func, lit};
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, Table, TableConfig, TableCreation, TableInsertion};
use indexlake_index_rstar::{RStarIndexKind, RStarIndexParams, WkbDialect};
use indexlake_integration_tests::data::prepare_simple_geom_table;
use indexlake_integration_tests::utils::{
    assert_data_file_count, assert_inline_row_count, table_scan,
};

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn create_rstar_index_on_existing_table(
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
    client.register_index_kind(Arc::new(RStarIndexKind));

    let table = prepare_simple_geom_table(&client, format).await?;
    let namespace_name = table.namespace_name.clone();
    let table_name = table.table_name.clone();

    let index_creation = IndexCreation {
        name: "rstar_index".to_string(),
        kind: RStarIndexKind.kind().to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        concurrency: 1,
        if_not_exists: false,
        config: Default::default(),
    };
    table.create_index(index_creation).await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let scan = TableScan::default().with_filters(vec![func(
        "intersects",
        vec![
            col("geom"),
            lit(Geometry::from(Point::new(11.0, 11.0)).to_wkb(CoordDimensions::xy())?),
        ],
        DataType::Boolean,
    )]);

    let table_str = table_scan(&table, scan).await?;
    println!("{}", table_str);
    assert_eq!(
        table_str,
        r#"+----+--------------------------------------------+
| id | geom                                       |
+----+--------------------------------------------+
| 2  | 010100000000000000000026400000000000002640 |
+----+--------------------------------------------+"#,
    );

    Ok(())
}

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[tokio::test(flavor = "multi_thread")]
async fn rstar_inexact_filter_is_rechecked(
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
    client.register_index_kind(Arc::new(RStarIndexKind));

    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("geom", DataType::Binary, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
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

    // Rows 1/2 and 4/5 each form an X inside a 2x2 square, so within a pair the
    // bounding boxes are identical and only an exact geometry check can tell
    // the lines apart. With 5 rows and a limit of 3, the dump task moves rows
    // 1-3 into a data file and leaves rows 4-5 inline, exercising both scan
    // paths.
    let lines = vec![
        (1, LineString::from(vec![(0.0, 0.0), (2.0, 2.0)])),
        (2, LineString::from(vec![(0.0, 2.0), (2.0, 0.0)])),
        (3, LineString::from(vec![(10.0, 10.0), (12.0, 12.0)])),
        (4, LineString::from(vec![(4.0, 0.0), (6.0, 2.0)])),
        (5, LineString::from(vec![(4.0, 2.0), (6.0, 0.0)])),
    ];
    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(
                lines.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            )),
            Arc::new(BinaryArray::from_iter_values(lines.iter().map(
                |(_, line)| {
                    Geometry::from(line.clone())
                        .to_wkb(CoordDimensions::xy())
                        .unwrap()
                },
            ))),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // Wait for the dump task to finish so both inline rows and data files are
    // indexed.
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;
    assert_inline_row_count(&table, |count| count > 0).await?;
    assert_data_file_count(&table, |count| count > 0).await?;

    let index_creation = IndexCreation {
        name: "rstar_index".to_string(),
        kind: RStarIndexKind.kind().to_string(),
        key_columns: vec!["geom".to_string()],
        params: Arc::new(RStarIndexParams {
            wkb_dialect: WkbDialect::Wkb,
        }),
        concurrency: 1,
        if_not_exists: false,
        config: Default::default(),
    };
    table.create_index(index_creation).await?;

    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    let table = client.load_table(&namespace_name, &table_name).await?;

    // Data-file rows: the bounding box of this segment lies inside the
    // bounding boxes of both rows 1 and 2, but it is strictly below both
    // diagonals.
    let no_match = LineString::from(vec![(0.2, 0.1), (0.6, 0.1)]);
    assert_eq!(scan_ids(&table, no_match).await?, Vec::<i32>::new());

    // This segment crosses row 1 exactly at (0.5, 0.5) and does not cross row 2.
    // Its bounding box still overlaps the bounding box of row 2.
    let match_row_1 = LineString::from(vec![(0.0, 0.5), (1.0, 0.5)]);
    assert_eq!(scan_ids(&table, match_row_1).await?, vec![1]);

    // Same checks against the inline rows 4 and 5.
    let no_match = LineString::from(vec![(4.2, 0.1), (4.6, 0.1)]);
    assert_eq!(scan_ids(&table, no_match).await?, Vec::<i32>::new());

    let match_row_4 = LineString::from(vec![(4.0, 0.5), (5.0, 0.5)]);
    assert_eq!(scan_ids(&table, match_row_4).await?, vec![4]);

    Ok(())
}

async fn scan_ids(
    table: &Table,
    geometry: LineString<f64>,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    // Project only row_id and id. The geometry is needed by the recheck but
    // must not leak into the output schema.
    let scan = TableScan::default()
        .with_projection(Some(vec![0, 1]))
        .with_filters(vec![func(
            "intersects",
            vec![
                col("geom"),
                lit(Geometry::from(geometry).to_wkb(CoordDimensions::xy())?),
            ],
            DataType::Boolean,
        )]);

    let batches = table.scan(scan).await?.try_collect::<Vec<_>>().await?;
    let mut ids = Vec::new();
    for batch in batches {
        assert!(batch.schema().index_of("geom").is_err());
        let id_index = batch.schema().index_of("id")?;
        let ids_array = batch
            .column(id_index)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        ids.extend(ids_array.values().iter().copied());
    }
    ids.sort_unstable();
    Ok(ids)
}
