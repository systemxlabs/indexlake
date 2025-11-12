mod file;
mod storage;

pub use file::*;
pub use storage::*;

use indexlake::storage::{EntryMode, FileMetadata};
use std::time::UNIX_EPOCH;

pub(crate) fn parse_std_fs_metadata(metadata: &std::fs::Metadata) -> FileMetadata {
    let size = metadata.len();
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
    FileMetadata {
        size,
        last_modified,
        mode,
    }
}
