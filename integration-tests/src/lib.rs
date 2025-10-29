pub mod data;
mod docker;
pub mod utils;

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use indexlake::catalog::{Catalog, CatalogDataType, CatalogSchema, Column};
use indexlake::storage::Storage;
use indexlake_catalog_postgres::PostgresCatalogBuilder;
use indexlake_catalog_sqlite::SqliteCatalog;
use indexlake_storage_fs::FsStorage;
use indexlake_storage_s3::S3Storage;
use opendal::services::S3Config;

use crate::docker::DockerCompose;

static ENV_LOGGER: OnceLock<()> = OnceLock::new();
static POSTGRES_DB: OnceLock<DockerCompose> = OnceLock::new();
static MINIO: OnceLock<DockerCompose> = OnceLock::new();

pub fn init_env_logger() {
    unsafe {
        std::env::set_var(
            "RUST_LOG",
            "info,indexlake=debug,indexlake_catalog_postgres=debug,indexlake_catalog_sqlite=debug,indexlake_index_rstar=debug,indexlake_index_bm25=debug",
        );
    }
    ENV_LOGGER.get_or_init(|| {
        env_logger::init();
    });
}

pub fn setup_sqlite_db() -> String {
    let db_path = format!(
        "{}/tmp/sqlite/{}.db",
        env!("CARGO_MANIFEST_DIR"),
        uuid::Uuid::new_v4()
    );
    std::fs::create_dir_all(PathBuf::from(&db_path).parent().unwrap()).unwrap();
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    conn.execute_batch(include_str!("../testdata/sqlite/init_catalog.sql"))
        .unwrap();
    db_path
}

pub fn setup_postgres_db() -> DockerCompose {
    let docker_compose = DockerCompose::new(
        "postgres",
        format!("{}/testdata/postgres", env!("CARGO_MANIFEST_DIR")),
    );
    docker_compose.up();
    docker_compose
}

pub fn setup_minio() -> DockerCompose {
    let docker_compose = DockerCompose::new(
        "minio",
        format!("{}/testdata/minio", env!("CARGO_MANIFEST_DIR")),
    );
    docker_compose.up();
    docker_compose
}

pub fn catalog_sqlite() -> Arc<dyn Catalog> {
    let db_path = setup_sqlite_db();
    Arc::new(SqliteCatalog::try_new(db_path).unwrap())
}

pub async fn catalog_postgres() -> Arc<dyn Catalog> {
    let _ = POSTGRES_DB.get_or_init(setup_postgres_db);
    let builder = PostgresCatalogBuilder::new("localhost", 5432, "postgres", "password")
        .dbname("postgres")
        .pool_max_size(50);
    let catalog = Arc::new(builder.build().await.unwrap());

    let schema = Arc::new(CatalogSchema::new(vec![Column::new(
        "1",
        CatalogDataType::Int64,
        false,
    )]));

    let mut retry_count = 0;
    while let Err(_e) = catalog.query("select 1", schema.clone()).await {
        std::thread::sleep(std::time::Duration::from_secs(1));
        retry_count += 1;
        if retry_count > 100 {
            panic!("Postgres catalog connectivity check failed");
        }
    }

    catalog
}

pub fn storage_fs() -> Arc<dyn Storage> {
    let home = format!("{}/{}", env!("CARGO_MANIFEST_DIR"), "tmp/fs_storage");
    let fs_storage = FsStorage::new(home.into());
    Arc::new(fs_storage)
}

pub async fn storage_s3() -> Arc<dyn Storage> {
    let _ = MINIO.get_or_init(setup_minio);
    let mut config = S3Config::default();
    config.endpoint = Some("http://127.0.0.1:9000".to_string());
    config.access_key_id = Some("admin".to_string());
    config.secret_access_key = Some("password".to_string());
    config.region = Some("us-east-1".to_string());
    config.disable_config_load = true;
    config.disable_ec2_metadata = true;

    let s3_storage = S3Storage::new(config, "indexlake".into());

    let mut retry_count = 0;
    while let Err(_e) = s3_storage.connectivity_check().await {
        std::thread::sleep(std::time::Duration::from_secs(1));
        retry_count += 1;
        if retry_count > 100 {
            panic!("S3 storage connectivity check failed");
        }
    }

    Arc::new(s3_storage)
}
