use bb8::Pool;
use bb8_postgres::PostgresConnectionManager;
use bb8_postgres::tokio_postgres::NoTls;
use indexlake::{ILError, ILResult};

use crate::PostgresCatalog;

pub struct PostgresCatalogBuilder {
    host: String,
    port: u16,
    user: String,
    password: String,
    dbname: Option<String>,
    pool_max_size: u32,
    pool_min_idle: Option<u32>,
}

impl PostgresCatalogBuilder {
    pub fn new(
        host: impl Into<String>,
        port: u16,
        user: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            port,
            user: user.into(),
            password: password.into(),
            dbname: None,
            pool_max_size: 100,
            pool_min_idle: Some(5),
        }
    }

    pub fn dbname(mut self, dbname: impl Into<String>) -> Self {
        self.dbname = Some(dbname.into());
        self
    }

    pub fn pool_max_size(mut self, pool_max_size: u32) -> Self {
        self.pool_max_size = pool_max_size;
        self
    }

    pub fn pool_min_idle(mut self, pool_min_idle: Option<u32>) -> Self {
        self.pool_min_idle = pool_min_idle;
        self
    }

    pub async fn build(self) -> ILResult<PostgresCatalog> {
        let mut config = bb8_postgres::tokio_postgres::config::Config::new();
        config
            .host(&self.host)
            .port(self.port)
            .user(&self.user)
            .password(&self.password);
        if let Some(dbname) = self.dbname {
            config.dbname(dbname);
        }
        let manager = PostgresConnectionManager::new(config, NoTls);
        let pool = Pool::builder()
            .max_size(self.pool_max_size)
            .min_idle(self.pool_min_idle)
            .build(manager)
            .await
            .map_err(|e| ILError::catalog(format!("failed to build postgres pool: {e}")))?;
        Ok(PostgresCatalog::new(pool))
    }
}
