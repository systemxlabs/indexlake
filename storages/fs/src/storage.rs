use std::path::PathBuf;

use indexlake::{
    ILError, ILResult,
    storage::{File, Storage},
};
use opendal::{Operator, layers::RetryLayer, services::FsConfig};

use crate::LocalFile;

#[derive(Debug)]
pub struct FsStorage {
    pub root: PathBuf,
}

impl FsStorage {
    pub fn new(root: PathBuf) -> Self {
        FsStorage { root }
    }

    pub fn new_operator(&self) -> ILResult<Operator> {
        let mut cfg = FsConfig::default();
        cfg.root = Some(self.root.to_string_lossy().to_string());
        let op = Operator::from_config(cfg)
            .map_err(|e| ILError::storage(format!("Failed to create fs operator: {e}")))?
            .layer(RetryLayer::new())
            .finish();
        Ok(op)
    }
}

#[async_trait::async_trait]
impl Storage for FsStorage {
    async fn create(&self, relative_path: &str) -> ILResult<Box<dyn File>> {
        let op = self.new_operator()?;
        let s3_file = LocalFile {
            op,
            relative_path: relative_path.to_string(),
        };
        Ok(Box::new(s3_file))
    }
    async fn open(&self, relative_path: &str) -> ILResult<Box<dyn File>> {
        let op = self.new_operator()?;
        let s3_file = LocalFile {
            op,
            relative_path: relative_path.to_string(),
        };
        Ok(Box::new(s3_file))
    }
    async fn delete(&self, relative_path: &str) -> ILResult<()> {
        let op = self.new_operator()?;
        op.delete(relative_path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to delete file {relative_path}, e: {e}")))
    }
    async fn exists(&self, relative_path: &str) -> ILResult<bool> {
        let op = self.new_operator()?;
        op.exists(relative_path).await.map_err(|e| {
            ILError::storage(format!(
                "Failed to check existence of file {relative_path}, e: {e}"
            ))
        })
    }
    async fn remove_dir_all(&self, relative_path: &str) -> ILResult<()> {
        let op = self.new_operator()?;
        let relative_path = if relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };
        op.remove_all(&relative_path).await.map_err(|e| {
            ILError::storage(format!(
                "Failed to remove directory {relative_path}, e: {e}"
            ))
        })
    }
}
