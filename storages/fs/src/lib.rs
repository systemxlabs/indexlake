mod file;
mod storage;

use std::{os::windows::fs::MetadataExt, time::UNIX_EPOCH};

pub use file::*;
pub use storage::*;

use indexlake::{
    ILResult,
    storage::{EntryMode, FileMetadata},
};

pub(crate) fn parse_std_fs_metadata(metadata: &std::fs::Metadata) -> ILResult<FileMetadata> {
    let size = metadata.file_size();
    let last_modified = if let Ok(modified) = metadata.modified() {
        Some(
            modified
                .duration_since(UNIX_EPOCH)
                .expect("Time went backwards")
                .as_millis() as i64,
        )
    } else {
        None
    };
    let mode = if metadata.is_dir() {
        EntryMode::Directory
    } else if metadata.is_file() {
        EntryMode::File
    } else {
        EntryMode::Unknown
    };
    Ok(FileMetadata {
        size,
        last_modified,
        mode,
    })
}
