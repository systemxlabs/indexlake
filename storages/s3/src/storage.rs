use indexlake::{
    ILError, ILResult,
    storage::{InputFile, OutputFile, Storage},
};
use opendal::{Configurator, Operator, layers::RetryLayer, services::S3Config};

use crate::{S3InputFile, S3OutputFile};

#[derive(Debug)]
pub struct S3Storage {
    pub config: S3Config,
    pub bucket: String,
}

impl S3Storage {
    pub fn new(config: S3Config, bucket: String) -> Self {
        S3Storage { config, bucket }
    }

    pub fn new_operator(&self) -> ILResult<Operator> {
        let builder = self.config.clone().into_builder().bucket(&self.bucket);
        let op = Operator::new(builder)
            .map_err(|e| ILError::storage(format!("Failed to create operator: {e}")))?
            .layer(RetryLayer::new())
            .finish();
        Ok(op)
    }

    pub async fn connectivity_check(&self) -> ILResult<()> {
        let op = self.new_operator()?;
        op.list("")
            .await
            .map_err(|e| ILError::storage(format!("Failed to list root path, e: {e}")))?;
        Ok(())
    }
}

#[async_trait::async_trait]
impl Storage for S3Storage {
    async fn create(&self, relative_path: &str) -> ILResult<Box<dyn OutputFile>> {
        let op = self.new_operator()?;
        let s3_file = S3OutputFile {
            op,
            relative_path: relative_path.to_string(),
            writer: None,
        };
        Ok(Box::new(s3_file))
    }

    async fn open(&self, relative_path: &str) -> ILResult<Box<dyn InputFile>> {
        let op = self.new_operator()?;
        let s3_file = S3InputFile {
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
