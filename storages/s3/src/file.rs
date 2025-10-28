use std::{fmt::Debug, ops::Range};

use bytes::Bytes;
use indexlake::{
    ILError, ILResult,
    storage::{File, FileMetadata},
};
use opendal::Operator;

pub struct S3File {
    pub op: Operator,
    pub relative_path: String,
    pub writer: Option<opendal::Writer>,
}

impl Debug for S3File {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3File")
            .field("op", &self.op)
            .field("relative_path", &self.relative_path)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl File for S3File {
    async fn metadata(&self) -> ILResult<FileMetadata> {
        let file_metadata = self.op.stat(&self.relative_path).await.map_err(|e| {
            ILError::storage(format!(
                "Failed to read stat of file {}, e: {e}",
                self.relative_path
            ))
        })?;
        Ok(FileMetadata {
            size: file_metadata.content_length(),
        })
    }

    async fn read(&self, range: Range<u64>) -> ILResult<Bytes> {
        let reader = self
            .op
            .reader(&self.relative_path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to create opendal reader: {e}")))?;
        let buffer = reader
            .read(range)
            .await
            .map_err(|e| ILError::storage(format!("Failed to read range: {e}")))?;
        Ok(buffer.to_bytes())
    }

    async fn write(&mut self, data: Bytes) -> ILResult<()> {
        if self.writer.is_none() {
            self.writer =
                Some(self.op.writer(&self.relative_path).await.map_err(|e| {
                    ILError::storage(format!("Failed to create opendal writer: {e}"))
                })?);
        }
        let writer = self.writer.as_mut().unwrap();
        writer
            .write(data)
            .await
            .map_err(|e| ILError::storage(format!("Failed to write data: {e}")))?;
        Ok(())
    }

    async fn close(&mut self) -> ILResult<()> {
        if let Some(writer) = self.writer.as_mut() {
            writer
                .close()
                .await
                .map_err(|e| ILError::storage(format!("Failed to close writer: {e}")))?;
        }
        Ok(())
    }
}
