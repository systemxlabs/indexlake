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
    pool_size: usize,
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
            pool_size: 10,
        }
    }

    pub fn dbname(mut self, dbname: impl Into<String>) -> Self {
        self.dbname = Some(dbname.into());
        self
    }

    pub fn pool_size(mut self, pool_size: usize) -> Self {
        self.pool_size = pool_size;
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
            .max_size(self.pool_size as u32)
            .build(manager)
            .await
            .map_err(|e| ILError::catalog(format!("failed to build postgres pool: {e}")))?;
        Ok(PostgresCatalog::new(pool))
    }
}
