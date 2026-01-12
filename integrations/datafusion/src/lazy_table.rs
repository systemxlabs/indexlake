use std::sync::Arc;

use indexlake::table::Table;
use indexlake::Client;
use tokio::sync::Mutex;

/// A lazy-loaded table holder containing client and table metadata.
/// The actual Table is loaded on first access if not provided upfront.
#[derive(Debug, Clone)]
pub struct LazyTable {
    pub client: Arc<Client>,
    pub namespace_name: String,
    pub table_name: String,
    table: Arc<Mutex<Option<Arc<Table>>>>,
}

impl LazyTable {
    /// Create a new LazyTable without a pre-loaded table.
    pub fn new(client: Arc<Client>, namespace_name: String, table_name: String) -> Self {
        Self {
            client,
            namespace_name,
            table_name,
            table: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a new LazyTable with a pre-loaded table.
    pub fn with_table(
        client: Arc<Client>,
        namespace_name: String,
        table_name: String,
        table: Arc<Table>,
    ) -> Self {
        Self {
            client,
            namespace_name,
            table_name,
            table: Arc::new(Mutex::new(Some(table))),
        }
    }

    /// Get the table, loading it lazily if not already loaded.
    pub async fn get_or_load(&self) -> Result<Arc<Table>, indexlake::ILError> {
        let mut guard = self.table.lock().await;
        if let Some(table) = guard.as_ref() {
            return Ok(table.clone());
        }

        let table = self
            .client
            .load_table(&self.namespace_name, &self.table_name)
            .await?;
        let table = Arc::new(table);
        *guard = Some(table.clone());
        Ok(table)
    }

    /// Get the inner table mutex for cloning to new instances.
    pub fn table_mutex(&self) -> Arc<Mutex<Option<Arc<Table>>>> {
        self.table.clone()
    }

    /// Create a new LazyTable with the same metadata but a different table mutex.
    pub fn with_table_mutex(
        client: Arc<Client>,
        namespace_name: String,
        table_name: String,
        table: Arc<Mutex<Option<Arc<Table>>>>,
    ) -> Self {
        Self {
            client,
            namespace_name,
            table_name,
            table,
        }
    }
}
