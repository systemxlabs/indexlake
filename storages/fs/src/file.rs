use std::ops::Range;

use bytes::Bytes;
use indexlake::{
    ILError, ILResult,
    storage::{FileMetadata, InputFile, OutputFile},
};
use opendal::Operator;

#[derive(Debug)]
pub struct LocalInputFile {
    pub op: Operator,
    pub relative_path: String,
}

#[async_trait::async_trait]
impl InputFile for LocalInputFile {
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
}

#[derive(Debug)]
pub struct LocalOutputFile {
    pub op: Operator,
    pub relative_path: String,
}

#[async_trait::async_trait]
impl OutputFile for LocalOutputFile {
    async fn write(&mut self, data: Bytes) -> ILResult<()> {
        let mut writer = self
            .op
            .writer(&self.relative_path)
            .await
            .map_err(|e| ILError::storage(format!("Failed to create opendal writer: {e}")))?;
        writer
            .write(data)
            .await
            .map_err(|e| ILError::storage(format!("Failed to write data: {e}")))?;
        writer
            .close()
            .await
            .map_err(|e| ILError::storage(format!("Failed to close writer: {e}")))?;
        Ok(())
    }

    async fn close(&mut self) -> ILResult<()> {
        Ok(())
    }
}
