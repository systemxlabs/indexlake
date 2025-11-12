use std::{io::SeekFrom, ops::Range};

use bytes::Bytes;
use indexlake::{
    ILError, ILResult,
    storage::{FileMetadata, InputFile, OutputFile},
};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

use crate::parse_std_fs_metadata;

#[derive(Debug)]
pub struct LocalInputFile {
    pub file: tokio::fs::File,
    pub relative_path: String,
}

#[async_trait::async_trait]
impl InputFile for LocalInputFile {
    async fn metadata(&self) -> ILResult<FileMetadata> {
        let metadata = self
            .file
            .metadata()
            .await
            .map_err(|e| ILError::storage(format!("Failed to get file metadata: {e}")))?;
        Ok(parse_std_fs_metadata(&metadata))
    }

    async fn read(&mut self, range: Range<u64>) -> ILResult<Bytes> {
        self.file
            .seek(SeekFrom::Start(range.start))
            .await
            .map_err(|e| {
                ILError::storage(format!("Failed to seek file {}: {e}", self.relative_path))
            })?;
        let mut buffer = vec![0; (range.end - range.start) as usize];
        self.file.read_exact(&mut buffer).await.map_err(|e| {
            ILError::storage(format!("Failed to read file {}: {e}", self.relative_path))
        })?;
        Ok(Bytes::from(buffer))
    }
}

#[derive(Debug)]
pub struct LocalOutputFile {
    pub file: tokio::fs::File,
    pub relative_path: String,
}

#[async_trait::async_trait]
impl OutputFile for LocalOutputFile {
    async fn write(&mut self, data: Bytes) -> ILResult<()> {
        self.file.write(&data).await.map_err(|e| {
            ILError::storage(format!("Failed to write file {}: {e}", self.relative_path))
        })?;
        Ok(())
    }

    async fn close(&mut self) -> ILResult<()> {
        self.file.flush().await.map_err(|e| {
            ILError::storage(format!("Failed to flush file {}: {e}", self.relative_path))
        })?;
        Ok(())
    }
}
