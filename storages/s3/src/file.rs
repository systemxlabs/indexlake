use std::{fmt::Debug, ops::Range};

use bytes::Bytes;
use indexlake::{
    ILError, ILResult,
    storage::{FileMetadata, InputFile, OutputFile},
};
use opendal::Operator;

#[derive(Debug)]
pub struct S3InputFile {
    pub op: Operator,
    pub relative_path: String,
    pub metadata: FileMetadata,
}

#[async_trait::async_trait]
impl InputFile for S3InputFile {
    async fn metadata(&self) -> ILResult<FileMetadata> {
        Ok(self.metadata.clone())
    }

    async fn read(&mut self, range: Range<u64>) -> ILResult<Bytes> {
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
    async fn read_ranges(&mut self, ranges: Vec<Range<u64>>) -> ILResult<Vec<Bytes>> {
        let reader = self
            .op
            .reader(&self.relative_path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to create opendal reader: {e}")))?;
        let buffers = reader
            .fetch(ranges)
            .await
            .map_err(|e| ILError::storage(format!("Failed to read ranges: {e}")))?;
        Ok(buffers.into_iter().map(|b| b.to_bytes()).collect())
    }
}

pub struct S3OutputFile {
    pub op: Operator,
    pub relative_path: String,
    pub writer: Option<opendal::Writer>,
}

impl Debug for S3OutputFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("S3OutputFile")
            .field("op", &self.op)
            .field("relative_path", &self.relative_path)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl OutputFile for S3OutputFile {
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
