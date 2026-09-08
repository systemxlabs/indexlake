use std::{
    io::{Read, Seek, SeekFrom},
    ops::Range,
};

use bytes::Bytes;
use indexlake::{
    ILError, ILResult,
    storage::{FileMetadata, InputFile, OutputFile},
};
use tokio::{io::AsyncWriteExt, task::spawn_blocking};

use crate::parse_std_fs_metadata;

#[derive(Debug)]
pub struct LocalInputFile {
    pub file: std::fs::File,
    pub relative_path: String,
}

fn read_range(file: &mut std::fs::File, relative_path: &str, range: Range<u64>) -> ILResult<Bytes> {
    if range.end < range.start {
        return Err(ILError::storage(format!(
            "Invalid read range [{}, {}) on file {relative_path}",
            range.start, range.end
        )));
    }
    file.seek(SeekFrom::Start(range.start))
        .map_err(|e| ILError::storage(format!("Failed to seek file {relative_path}: {e}")))?;
    let mut buffer = vec![0; (range.end - range.start) as usize];
    file.read_exact(&mut buffer)
        .map_err(|e| ILError::storage(format!("Failed to read file {relative_path}: {e}")))?;
    Ok(Bytes::from(buffer))
}

#[async_trait::async_trait]
impl InputFile for LocalInputFile {
    async fn metadata(&self) -> ILResult<FileMetadata> {
        // fstat is a cheap non-blocking syscall, no need to offload it.
        let metadata = self.file.metadata().map_err(|e| {
            ILError::storage(format!(
                "Failed to get file metadata {}: {e}",
                self.relative_path
            ))
        })?;
        Ok(parse_std_fs_metadata(&metadata))
    }

    async fn read(&mut self, range: Range<u64>) -> ILResult<Bytes> {
        // try_clone dups the fd and clones share the file cursor; that is
        // safe because accesses to one LocalInputFile are serialized
        // (`&mut self` at the trait boundary).
        let mut file = self.file.try_clone().map_err(|e| {
            ILError::storage(format!(
                "Failed to clone file handle {}: {e}",
                self.relative_path
            ))
        })?;
        let relative_path = self.relative_path.clone();
        spawn_blocking(move || read_range(&mut file, &relative_path, range))
            .await
            .map_err(|e| ILError::storage(format!("Failed to join blocking read task: {e}")))?
    }

    async fn read_ranges(&mut self, ranges: Vec<Range<u64>>) -> ILResult<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        if ranges.len() == 1 {
            return self.read(ranges[0].clone()).await.map(|b| vec![b]);
        }
        // Offload all range reads into a single blocking task so the whole
        // batch costs one runtime dispatch instead of two per range. The
        // seeks and reads run serialized inside this one task, which is what
        // keeps the shared file cursor safe (see `read`).
        let mut file = self.file.try_clone().map_err(|e| {
            ILError::storage(format!(
                "Failed to clone file handle {}: {e}",
                self.relative_path
            ))
        })?;
        let relative_path = self.relative_path.clone();
        spawn_blocking(move || {
            ranges
                .into_iter()
                .map(|range| read_range(&mut file, &relative_path, range))
                .collect()
        })
        .await
        .map_err(|e| ILError::storage(format!("Failed to join blocking read task: {e}")))?
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
