use std::path::PathBuf;

use futures::StreamExt;
use indexlake::{
    ILError, ILResult,
    storage::{DirEntry, DirEntryStream, InputFile, OutputFile, Storage},
};
use tokio::fs::File;
use tokio_stream::wrappers::ReadDirStream;

use crate::{LocalInputFile, LocalOutputFile, parse_std_fs_metadata};

#[derive(Debug)]
pub struct FsStorage {
    pub root: PathBuf,
}

impl FsStorage {
    pub fn new(root: PathBuf) -> Self {
        FsStorage { root }
    }

    pub fn absolute_path(&self, relative_path: &str) -> PathBuf {
        self.root.join(relative_path)
    }
}

#[async_trait::async_trait]
impl Storage for FsStorage {
    async fn create(&self, relative_path: &str) -> ILResult<Box<dyn OutputFile>> {
        let path = self.absolute_path(relative_path);
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                ILError::storage(format!(
                    "Failed to create directory for {relative_path}: {e}"
                ))
            })?;
        }
        let file = File::create(path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to create file {relative_path}: {e}")))?;
        let file = LocalOutputFile {
            file,
            relative_path: relative_path.to_string(),
        };
        Ok(Box::new(file))
    }

    async fn open(&self, relative_path: &str) -> ILResult<Box<dyn InputFile>> {
        let path = self.absolute_path(relative_path);
        let file = File::open(path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to open file {relative_path}: {e}")))?;
        let file = LocalInputFile {
            file,
            relative_path: relative_path.to_string(),
        };
        Ok(Box::new(file))
    }

    async fn delete(&self, relative_path: &str) -> ILResult<()> {
        let path = self.absolute_path(relative_path);
        tokio::fs::remove_file(path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to remove file {relative_path}: {e}")))
    }

    async fn exists(&self, relative_path: &str) -> ILResult<bool> {
        let path = self.absolute_path(relative_path);
        tokio::fs::try_exists(path).await.map_err(|e| {
            ILError::storage(format!(
                "Failed to check file existing {relative_path}: {e}"
            ))
        })
    }

    async fn list(&self, relative_path: &str) -> ILResult<DirEntryStream> {
        let path = self.absolute_path(relative_path);
        let read_dir = tokio::fs::read_dir(path).await.map_err(|e| {
            ILError::storage(format!("Failed to read directory {relative_path}: {e}"))
        })?;
        let stream = ReadDirStream::new(read_dir);
        let stream = stream
            .then(|entry| async move {
                let entry =
                    entry.map_err(|e| ILError::storage(format!("Failed to read entry: {e}")))?;
                let metadata = entry.metadata().await.map_err(|e| {
                    ILError::storage(format!("Failed to read dir entry metadata: {e}"))
                })?;
                let dir_entry = DirEntry {
                    name: entry.file_name().to_string_lossy().into_owned(),
                    metadata: parse_std_fs_metadata(&metadata)?,
                };
                Ok(dir_entry)
            })
            .boxed();

        Ok(stream)
    }

    async fn remove_dir_all(&self, _relative_path: &str) -> ILResult<()> {
        todo!()
    }
}
