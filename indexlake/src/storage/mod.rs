mod parquet;

use arrow::array::FixedSizeBinaryArray;
use bytes::Bytes;
pub(crate) use parquet::*;
use uuid::Uuid;

use crate::catalog::DataFileRecord;
use crate::expr::{Expr, row_ids_in_list_expr};
use crate::{ILError, ILResult, RecordBatchStream};
use arrow_schema::Schema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::Range;

#[async_trait::async_trait]
pub trait File: Debug + Send + Sync + 'static {
    async fn metadata(&self) -> ILResult<FileMetadata>;
    async fn read(&self, range: Range<u64>) -> ILResult<Bytes>;
    async fn write(&mut self, bs: Bytes) -> ILResult<()>;
    async fn close(&mut self) -> ILResult<()>;
}

#[derive(Debug)]
pub struct FileMetadata {
    pub size: u64,
}

#[async_trait::async_trait]
impl File for Box<dyn File> {
    async fn metadata(&self) -> ILResult<FileMetadata> {
        self.as_ref().metadata().await
    }
    async fn read(&self, range: Range<u64>) -> ILResult<Bytes> {
        self.as_ref().read(range).await
    }
    async fn write(&mut self, bs: Bytes) -> ILResult<()> {
        self.as_mut().write(bs).await
    }
    async fn close(&mut self) -> ILResult<()> {
        self.as_mut().close().await
    }
}

#[async_trait::async_trait]
pub trait Storage: Debug + Send + Sync + 'static {
    async fn create(&self, relative_path: &str) -> ILResult<Box<dyn File>>;
    async fn open(&self, relative_path: &str) -> ILResult<Box<dyn File>>;
    async fn delete(&self, relative_path: &str) -> ILResult<()>;
    async fn exists(&self, relative_path: &str) -> ILResult<bool>;
    async fn remove_dir_all(&self, relative_path: &str) -> ILResult<()>;
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
    storage: &dyn Storage,
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
    storage: &dyn Storage,
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
    storage: &dyn Storage,
    relative_path: &str,
    format: DataFileFormat,
) -> ILResult<FixedSizeBinaryArray> {
    match format {
        DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => {
            read_row_id_array_from_parquet(storage, relative_path).await
        }
    }
}
