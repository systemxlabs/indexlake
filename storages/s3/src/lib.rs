mod file;
mod storage;

pub use file::*;
pub use opendal::services::S3Config;
pub use storage::*;

use indexlake::{
    ILError, ILResult,
    storage::{EntryMode, FileMetadata},
};

pub(crate) fn parse_opendal_metadata(metadata: &opendal::Metadata) -> ILResult<FileMetadata> {
    Ok(FileMetadata {
        size: metadata.content_length(),
        mode: parse_opendal_entry_mode(metadata.mode())?,
        last_modified: metadata.last_modified().map(|dt| dt.timestamp_millis()),
    })
}

pub(crate) fn parse_opendal_entry_mode(mode: opendal::EntryMode) -> ILResult<EntryMode> {
    match mode {
        opendal::EntryMode::DIR => Ok(EntryMode::Directory),
        opendal::EntryMode::FILE => Ok(EntryMode::File),
        opendal::EntryMode::Unknown => {
            Err(ILError::storage("Unrecognized opendal entry mode: {mode}"))
        }
    }
}
