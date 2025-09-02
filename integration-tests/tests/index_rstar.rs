use arrow::datatypes::DataType;
use indexlake::{Client, catalog::Catalog, index::IndexKind, storage::Storage, table::TableScan};
use indexlake_integration_tests::{
    catalog_postgres, catalog_sqlite, init_env_logger, storage_fs, storage_s3,
};
use std::sync::Arc;

use geo::{Geometry, Point};
use geozero::{CoordDimensions, ToWkb};
use indexlake::expr::{col, func, lit};
use indexlake::storage::DataFileFormat;
use indexlake::table::IndexCreation;
use indexlake_index_rstar::{RStarIndexKind, RStarIndexParams, WkbDialect};
use indexlake_integration_tests::data::prepare_simple_geom_table;
use indexlake_integration_tests::utils::table_scan;

#[rstest::rstest]
#[case(async { catalog_sqlite() }, async { storage_fs() }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV1)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::ParquetV2)]
#[case(async { catalog_postgres().await }, async { storage_s3().await }, DataFileFormat::LanceV2_0)]
#[tokio::test(flavor = "multi_thread")]
async fn create_rstar_index_on_existing_table(
    #[future(awt)]
    #[case]
    catalog: Arc<dyn Catalog>,
    #[future(awt)]
    #[case]
    storage: Arc<Storage>,
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
        if_not_exists: false,
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
