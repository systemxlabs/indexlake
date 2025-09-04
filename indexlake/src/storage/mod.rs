mod fs;
#[cfg(feature = "lance-format")]
mod lance;
mod parquet;
mod s3;

#[cfg(feature = "lance-format")]
pub(crate) use lance::*;
pub use opendal::services::S3Config;
pub(crate) use parquet::*;
use uuid::Uuid;

use crate::catalog::DataFileRecord;
use crate::expr::Expr;
use crate::storage::fs::FsStorage;
use crate::storage::s3::S3Storage;
use crate::{ILError, ILResult, RecordBatchStream};
use arrow_schema::Schema;
use opendal::Operator;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

    pub fn root_path(&self) -> ILResult<String> {
        match self {
            Storage::Fs(fs) => Ok(fs.root.to_string_lossy().to_string()),
            Storage::S3(s3) => Ok(format!("s3://{}/", s3.bucket)),
        }
    }

    pub fn storage_options(&self) -> ILResult<HashMap<String, String>> {
        match self {
            Storage::Fs(_fs) => Ok(HashMap::new()),
            Storage::S3(s3) => {
                let mut options = HashMap::new();
                options.insert(
                    "aws_access_key_id".to_string(),
                    s3.config
                        .access_key_id
                        .clone()
                        .ok_or_else(|| ILError::internal("Access key id is not set"))?,
                );
                options.insert(
                    "aws_secret_access_key".to_string(),
                    s3.config
                        .secret_access_key
                        .clone()
                        .ok_or_else(|| ILError::internal("Secret access key is not set"))?,
                );
                options.insert(
                    "aws_endpoint".to_string(),
                    s3.config
                        .endpoint
                        .clone()
                        .ok_or_else(|| ILError::internal("Endpoint is not set"))?,
                );
                options.insert("aws_allow_http".to_string(), "true".to_string());
                options.insert(
                    "aws_region".to_string(),
                    s3.config
                        .region
                        .clone()
                        .ok_or_else(|| ILError::internal("Region is not set"))?,
                );
                options.insert("AWS_EC2_METADATA_DISABLED".to_string(), "true".to_string());
                options.insert("AWS_S3_ALLOW_UNSAFE_RENAME".to_string(), "true".to_string());
                Ok(options)
            }
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
    LanceV2_0,
}

impl std::fmt::Display for DataFileFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataFileFormat::ParquetV1 => write!(f, "ParquetV1"),
            DataFileFormat::ParquetV2 => write!(f, "ParquetV2"),
            DataFileFormat::LanceV2_0 => write!(f, "LanceV2_0"),
        }
    }
}

impl std::str::FromStr for DataFileFormat {
    type Err = ILError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "ParquetV1" => Ok(DataFileFormat::ParquetV1),
            "ParquetV2" => Ok(DataFileFormat::ParquetV2),
            "LanceV2_0" => Ok(DataFileFormat::LanceV2_0),
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
    filters: Vec<Expr>,
    row_ids: Option<&HashSet<Uuid>>,
    batch_size: usize,
) -> ILResult<RecordBatchStream> {
    match data_file_record.format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            read_parquet_file_by_record(
                storage,
                table_schema,
                data_file_record,
                projection,
                filters,
                row_ids,
                batch_size,
            )
            .await
        }
        DataFileFormat::LanceV2_0 => {
            #[cfg(feature = "lance-format")]
            {
                read_lance_file_by_record(
                    storage,
                    table_schema,
                    data_file_record,
                    projection,
                    filters,
                    row_ids,
                    batch_size,
                )
                .await
            }
            #[cfg(not(feature = "lance-format"))]
            Err(ILError::not_supported(
                "Lance format feature is not enabled",
            ))
        }
    }
}

pub(crate) async fn read_data_file_by_record_and_row_id_condition(
    storage: &Storage,
    table_schema: &Schema,
    data_file_record: &DataFileRecord,
    projection: Option<Vec<usize>>,
    row_id_condition: &Expr,
) -> ILResult<RecordBatchStream> {
    match data_file_record.format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            read_parquet_file_by_record_and_row_id_condition(
                storage,
                table_schema,
                data_file_record,
                projection,
                row_id_condition,
            )
            .await
        }
        DataFileFormat::LanceV2_0 => {
            #[cfg(feature = "lance-format")]
            {
                read_lance_file_by_record_and_row_id_condition(
                    storage,
                    table_schema,
                    data_file_record,
                    projection,
                    row_id_condition,
                )
                .await
            }
            #[cfg(not(feature = "lance-format"))]
            Err(ILError::not_supported(
                "Lance format feature is not enabled",
            ))
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
        DataFileFormat::LanceV2_0 => {
            #[cfg(feature = "lance-format")]
            {
                find_matched_row_ids_from_lance_file(
                    storage,
                    table_schema,
                    condition,
                    data_file_record,
                )
                .await
            }
            #[cfg(not(feature = "lance-format"))]
            Err(ILError::not_supported(
                "Lance format feature is not enabled",
            ))
        }
    }
}
