mod fs;
mod parquet;
mod s3;

use arrow::array::FixedSizeBinaryArray;
pub use opendal::services::S3Config;
pub(crate) use parquet::*;
use uuid::Uuid;

use crate::catalog::DataFileRecord;
use crate::expr::{Expr, row_ids_in_list_expr};
use crate::storage::fs::FsStorage;
use crate::storage::s3::S3Storage;
use crate::{ILError, ILResult, RecordBatchStream};
use arrow_schema::Schema;
use opendal::Operator;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum Storage {
    Fs(FsStorage),
    S3(Box<S3Storage>),
}

impl Storage {
    pub fn new_fs(root: impl Into<PathBuf>) -> Self {
        Storage::Fs(FsStorage::new(root.into()))
    }

    pub fn new_s3(config: S3Config, bucket: impl Into<String>) -> Self {
        Storage::S3(Box::new(S3Storage::new(config, bucket.into())))
    }

    pub async fn delete(&self, relative_path: &str) -> ILResult<()> {
        let op = self.new_operator()?;
        Ok(op.delete(relative_path).await?)
    }

    pub async fn remove_dir_all(&self, relative_path: &str) -> ILResult<()> {
        let op = self.new_operator()?;
        let relative_path = if relative_path.ends_with('/') {
            relative_path.to_string()
        } else {
            format!("{relative_path}/")
        };
        Ok(op.remove_all(&relative_path).await?)
    }

    pub async fn exists(&self, relative_path: &str) -> ILResult<bool> {
        let op = self.new_operator()?;
        Ok(op.exists(relative_path).await?)
    }

    pub(crate) fn new_operator(&self) -> ILResult<Operator> {
        match self {
            Storage::Fs(fs) => fs.new_operator(),
            Storage::S3(s3) => s3.new_operator(),
        }
    }

    pub async fn create_file(&self, relative_path: &str) -> ILResult<OutputFile> {
        let op = self.new_operator()?;
        let writer = op.writer(relative_path).await?;
        Ok(OutputFile {
            op,
            relative_path: relative_path.to_string(),
            writer,
        })
    }

    pub async fn open_file(&self, relative_path: &str) -> ILResult<InputFile> {
        let op = self.new_operator()?;
        let reader = op.reader(relative_path).await?;
        Ok(InputFile {
            op,
            relative_path: relative_path.to_string(),
            reader,
        })
    }

    pub async fn connectivity_check(&self) -> ILResult<()> {
        let op = self.new_operator()?;
        op.list("").await?;
        Ok(())
    }
}

/// Output file is used for writing to files.
pub struct OutputFile {
    op: Operator,
    relative_path: String,
    writer: opendal::Writer,
}

impl OutputFile {
    pub async fn file_size_bytes(&self) -> ILResult<u64> {
        let meta = self.op.stat(&self.relative_path).await?;
        Ok(meta.content_length())
    }

    pub async fn delete(&self) -> ILResult<()> {
        Ok(self.op.delete(&self.relative_path).await?)
    }

    pub fn writer(&mut self) -> &mut opendal::Writer {
        &mut self.writer
    }

    pub async fn close(&mut self) -> ILResult<()> {
        self.writer.close().await?;
        Ok(())
    }
}

pub struct InputFile {
    op: Operator,
    relative_path: String,
    reader: opendal::Reader,
}

impl InputFile {
    pub async fn file_size_bytes(&self) -> ILResult<u64> {
        let meta = self.op.stat(&self.relative_path).await?;
        Ok(meta.content_length())
    }

    pub async fn delete(&self) -> ILResult<()> {
        Ok(self.op.delete(&self.relative_path).await?)
    }

    pub async fn read(&self) -> ILResult<bytes::Bytes> {
        Ok(self.op.read(&self.relative_path).await?.to_bytes())
    }

    pub fn reader(&self) -> &opendal::Reader {
        &self.reader
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DataFileFormat {
    ParquetV1,
    ParquetV2,
}

impl std::fmt::Display for DataFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFileFormat::ParquetV1 => write!(f, "ParquetV1"),
            DataFileFormat::ParquetV2 => write!(f, "ParquetV2"),
        }
    }
}

impl std::str::FromStr for DataFileFormat {
    type Err = ILError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ParquetV1" => Ok(DataFileFormat::ParquetV1),
            "ParquetV2" => Ok(DataFileFormat::ParquetV2),
            _ => Err(ILError::invalid_input(format!(
                "Invalid data file format: {s}"
            ))),
        }
    }
}

pub(crate) async fn read_data_file_by_record(
    storage: &Storage,
    table_schema: &Schema,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    mut filters: Vec<Expr>,
    row_ids: Option<Vec<Uuid>>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    if let Some(row_ids) = row_ids {
        let row_id_filter = row_ids_in_list_expr(row_ids);
        filters.push(row_id_filter);
    }
    match data_file_record.format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            read_parquet_file_by_record(
                storage,
                table_schema,
                data_file_record,
                projection,
                filters,
                batch_size,
            )
            .await
        }
    }
}

pub(crate) async fn find_matched_row_ids_from_data_file(
    storage: &Storage,
    table_schema: &Schema,
    condition: &Expr,
    data_file_record: &DataFileRecord,
) -> ILResult<HashSet<Uuid>> {
    match data_file_record.format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            find_matched_row_ids_from_parquet_file(
                storage,
                table_schema,
                condition,
                data_file_record,
            )
            .await
        }
    }
}

pub(crate) async fn read_row_id_array_from_data_file(
    storage: &Storage,
    relative_path: &str,
    format: DataFileFormat,
) -> ILResult<FixedSizeBinaryArray> {
    match format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            read_row_id_array_from_parquet(storage, relative_path).await
        }
    }
}
