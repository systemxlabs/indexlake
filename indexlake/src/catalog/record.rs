use std::collections::HashMap;
use std::ops::Range;

use arrow::datatypes::{DataType, Field};
use parquet::arrow::arrow_reader::RowSelection;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::catalog::{
    CatalogDataType, CatalogDatabase, CatalogSchema, Column, Row, Scalar, deserialize_scalar,
    serialize_scalar,
};
use crate::storage::DataFileFormat;
use crate::table::TableConfig;
use crate::{ILError, ILResult};

#[derive(Debug, Clone)]
pub(crate) struct TableRecord {
    pub(crate) table_id: Uuid,
    pub(crate) table_name: String,
    pub(crate) namespace_id: Uuid,
    pub(crate) config: TableConfig,
    pub(crate) schema_metadata: HashMap<String, String>,
}

impl TableRecord {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        let config_str = serde_json::to_string(&self.config)
            .map_err(|e| ILError::internal(format!("Failed to serialize table config: {e:?}")))?;
        let schema_metadata_str = serde_json::to_string(&self.schema_metadata).map_err(|e| {
            ILError::internal(format!("Failed to serialize table schema metadata: {e:?}"))
        })?;
        Ok(format!(
            "({}, '{}', {}, '{}', '{}')",
            database.sql_uuid_literal(&self.table_id),
            self.table_name,
            database.sql_uuid_literal(&self.namespace_id),
            config_str,
            schema_metadata_str,
        ))
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("table_id", CatalogDataType::Uuid, false),
            Column::new("table_name", CatalogDataType::Utf8, false),
            Column::new("namespace_id", CatalogDataType::Uuid, false),
            Column::new("config", CatalogDataType::Utf8, false),
            Column::new("schema_metadata", CatalogDataType::Utf8, false),
        ])
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let table_id = row.uuid(0)?.expect("table_id is not null");
        let table_name = row.utf8_owned(1)?.expect("table_name is not null");
        let namespace_id = row.uuid(2)?.expect("namespace_id is not null");
        let config_str = row.utf8(3)?.expect("config is not null");
        let config: TableConfig = serde_json::from_str(config_str)
            .map_err(|e| ILError::internal(format!("Failed to deserialize table config: {e:?}")))?;
        let schema_metadata_str = row.utf8(4)?.expect("schema_metadata is not null");
        let schema_metadata: HashMap<String, String> = serde_json::from_str(schema_metadata_str)
            .map_err(|e| {
                ILError::internal(format!(
                    "Failed to deserialize table schema metadata: {e:?}"
                ))
            })?;
        Ok(TableRecord {
            table_id,
            table_name,
            namespace_id,
            config,
            schema_metadata,
        })
    }
}

#[derive(Debug, Clone)]
pub struct FieldRecord {
    pub field_id: Uuid,
    pub table_id: Uuid,
    pub field_name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default_value: Option<Scalar>,
    pub metadata: HashMap<String, String>,
}

impl FieldRecord {
    pub(crate) fn new(
        field_id: Uuid,
        table_id: Uuid,
        field: &Field,
        default_value: Option<Scalar>,
    ) -> Self {
        Self {
            field_id,
            table_id,
            field_name: field.name().to_string(),
            data_type: field.data_type().clone(),
            nullable: field.is_nullable(),
            default_value,
            metadata: field.metadata().clone(),
        }
    }

    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> ILResult<String> {
        let data_type_str = serde_json::to_string(&self.data_type)
            .map_err(|e| ILError::internal(format!("Failed to serialize data type: {e:?}")))?;
        let default_value_sql = match self.default_value.as_ref() {
            Some(value) => {
                let bytes = serialize_scalar(value)?;
                database.sql_binary_literal(&bytes)
            }
            None => "null".to_string(),
        };
        let metadata_str = serde_json::to_string(&self.metadata)
            .map_err(|e| ILError::internal(format!("Failed to serialize field metadata: {e:?}")))?;
        Ok(format!(
            "({}, {}, '{}', '{}', {}, {}, '{}')",
            database.sql_uuid_literal(&self.field_id),
            database.sql_uuid_literal(&self.table_id),
            self.field_name,
            data_type_str,
            self.nullable,
            default_value_sql,
            metadata_str
        ))
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("field_id", CatalogDataType::Uuid, false),
            Column::new("table_id", CatalogDataType::Uuid, false),
            Column::new("field_name", CatalogDataType::Utf8, false),
            Column::new("data_type", CatalogDataType::Utf8, false),
            Column::new("nullable", CatalogDataType::Boolean, false),
            Column::new("default_value", CatalogDataType::Binary, true),
            Column::new("metadata", CatalogDataType::Utf8, false),
        ])
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let field_id = row.uuid(0)?.expect("field_id is not null");
        let table_id = row.uuid(1)?.expect("table_id is not null");
        let field_name = row.utf8_owned(2)?.expect("field_name is not null");
        let data_type_str = row.utf8(3)?.expect("data_type is not null");
        let data_type: DataType = serde_json::from_str(data_type_str)
            .map_err(|e| ILError::internal(format!("Failed to deserialize data type: {e:?}")))?;
        let nullable = row.boolean(4)?.expect("nullable is not null");
        let default_value_bytes = row.binary(5)?;
        let default_value = match default_value_bytes {
            Some(bytes) => Some(deserialize_scalar(bytes)?),
            None => None,
        };

        let metadata_str = row.utf8(6)?.expect("metadata is not null");
        let metadata: HashMap<String, String> =
            serde_json::from_str(metadata_str).map_err(|e| {
                ILError::internal(format!("Failed to deserialize field metadata: {e:?}"))
            })?;
        Ok(FieldRecord {
            field_id,
            table_id,
            field_name,
            data_type,
            nullable,
            default_value,
            metadata,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileRecord {
    pub data_file_id: Uuid,
    pub table_id: Uuid,
    pub format: DataFileFormat,
    pub relative_path: String,
    pub record_count: i64,
    pub validity: RowValidity,
}

impl DataFileRecord {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> String {
        let validity_bytes = self.validity.bytes();
        format!(
            "({}, {}, '{}', '{}', {}, {})",
            database.sql_uuid_literal(&self.data_file_id),
            database.sql_uuid_literal(&self.table_id),
            self.format,
            self.relative_path,
            self.record_count,
            database.sql_binary_literal(validity_bytes),
        )
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("data_file_id", CatalogDataType::Uuid, false),
            Column::new("table_id", CatalogDataType::Uuid, false),
            Column::new("format", CatalogDataType::Utf8, false),
            Column::new("relative_path", CatalogDataType::Utf8, false),
            Column::new("record_count", CatalogDataType::Int64, false),
            Column::new("validity", CatalogDataType::Binary, false),
        ])
    }

    pub(crate) fn build_relative_path(
        namespace_id: &Uuid,
        table_id: &Uuid,
        data_file_id: &Uuid,
        format: DataFileFormat,
    ) -> String {
        format!(
            "{namespace_id}/{table_id}/{data_file_id}.{}",
            match format {
                DataFileFormat::ParquetV1 | DataFileFormat::ParquetV2 => "parquet",
            }
        )
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let data_file_id = row.uuid(0)?.expect("data_file_id is not null");
        let table_id = row.uuid(1)?.expect("table_id is not null");
        let format = row
            .utf8(2)?
            .expect("format is not null")
            .parse::<DataFileFormat>()
            .map_err(|e| ILError::internal(format!("Failed to parse data file format: {e:?}")))?;
        let relative_path = row.utf8_owned(3)?.expect("relative_path is not null");
        let record_count = row.int64(4)?.expect("record_count is not null");

        let validity_bytes = row.binary_owned(5)?.expect("validity is not null");
        let validity = RowValidity::from(validity_bytes, record_count as usize);

        Ok(DataFileRecord {
            data_file_id,
            table_id,
            format,
            relative_path,
            record_count,
            validity,
        })
    }

    pub(crate) fn valid_row_count(&self) -> usize {
        self.validity.iter().filter(|valid| *valid).count()
    }

    pub(crate) fn row_ranges(&self) -> Vec<Range<usize>> {
        let offsets = self
            .validity
            .iter()
            .enumerate()
            .filter(|(_, valid)| *valid)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();

        let mut ranges = Vec::new();
        let mut offset_idx = 0;
        while offset_idx < offsets.len() {
            let current_offset = offsets[offset_idx];
            let mut next_offset_idx = offset_idx + 1;
            while next_offset_idx < offsets.len()
                && offsets[next_offset_idx] == current_offset + (next_offset_idx - offset_idx)
            {
                next_offset_idx += 1;
            }
            ranges.push(current_offset..offsets[next_offset_idx - 1] + 1);
            offset_idx = next_offset_idx;
        }
        ranges
    }

    pub(crate) fn row_selection(&self) -> RowSelection {
        let ranges = self.row_ranges();
        RowSelection::from_consecutive_ranges(ranges.into_iter(), self.record_count as usize)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RowValidity {
    pub(crate) validity: Vec<u8>,
    pub(crate) num_rows: usize,
}

impl RowValidity {
    pub fn new(num_rows: usize) -> Self {
        let num_bytes = num_rows.div_ceil(8);
        let validity = vec![u8::MAX; num_bytes];
        Self { validity, num_rows }
    }

    pub fn from(bytes: Vec<u8>, num_rows: usize) -> Self {
        assert_eq!(bytes.len(), num_rows.div_ceil(8));
        Self {
            validity: bytes,
            num_rows,
        }
    }

    pub fn set(&mut self, row_idx: usize, valid: bool) {
        assert!(row_idx < self.num_rows);

        let byte_idx = row_idx / 8;
        let bit_idx = row_idx % 8;

        if valid {
            self.validity[byte_idx] |= 1 << bit_idx;
        } else {
            self.validity[byte_idx] &= !(1 << bit_idx);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.validity
            .iter()
            .enumerate()
            .flat_map(|(byte_idx, byte)| {
                let bit_len = if byte_idx == self.validity.len() - 1 {
                    self.num_rows % 8
                } else {
                    8
                };
                (0..bit_len).map(move |bit| (byte & (1 << bit)) != 0)
            })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.validity
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IndexRecord {
    pub(crate) index_id: Uuid,
    pub(crate) table_id: Uuid,
    pub(crate) index_name: String,
    pub(crate) index_kind: String,
    pub(crate) key_field_ids: Vec<Uuid>,
    pub(crate) params: String,
}

impl IndexRecord {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> String {
        let key_field_ids_str = self
            .key_field_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!(
            "({}, {}, '{}', '{}', '{}', '{}')",
            database.sql_uuid_literal(&self.index_id),
            database.sql_uuid_literal(&self.table_id),
            self.index_name,
            self.index_kind,
            key_field_ids_str,
            self.params
        )
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("index_id", CatalogDataType::Uuid, false),
            Column::new("table_id", CatalogDataType::Uuid, false),
            Column::new("index_name", CatalogDataType::Utf8, false),
            Column::new("index_kind", CatalogDataType::Utf8, false),
            Column::new("key_field_ids", CatalogDataType::Utf8, false),
            Column::new("params", CatalogDataType::Utf8, false),
        ])
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let index_id = row.uuid(0)?.expect("index_id is not null");
        let table_id = row.uuid(1)?.expect("table_id is not null");
        let index_name = row.utf8_owned(2)?.expect("index_name is not null");
        let index_kind = row.utf8_owned(3)?.expect("kind is not null");
        let key_field_ids_str = row.utf8(4)?.expect("key_field_ids is not null");
        let key_field_ids = key_field_ids_str
            .split(",")
            .map(Uuid::parse_str)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| ILError::internal(format!("Failed to parse key field ids: {e:?}")))?;
        let params = row.utf8_owned(5)?.expect("params is not null");
        Ok(IndexRecord {
            index_id,
            index_name,
            index_kind,
            table_id,
            key_field_ids,
            params,
        })
    }
}

pub(crate) struct IndexFileRecord {
    pub(crate) index_file_id: Uuid,
    pub(crate) table_id: Uuid,
    pub(crate) index_id: Uuid,
    pub(crate) data_file_id: Uuid,
    pub(crate) relative_path: String,
}

impl IndexFileRecord {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> String {
        format!(
            "({}, {}, {}, {}, '{}')",
            database.sql_uuid_literal(&self.index_file_id),
            database.sql_uuid_literal(&self.table_id),
            database.sql_uuid_literal(&self.index_id),
            database.sql_uuid_literal(&self.data_file_id),
            self.relative_path
        )
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("index_file_id", CatalogDataType::Uuid, false),
            Column::new("table_id", CatalogDataType::Uuid, false),
            Column::new("index_id", CatalogDataType::Uuid, false),
            Column::new("data_file_id", CatalogDataType::Uuid, false),
            Column::new("relative_path", CatalogDataType::Utf8, false),
        ])
    }

    pub(crate) fn build_relative_path(
        namespace_id: &Uuid,
        table_id: &Uuid,
        index_file_id: &Uuid,
    ) -> String {
        format!("{namespace_id}/{table_id}/{index_file_id}.index")
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let index_file_id = row.uuid(0)?.expect("index_file_id is not null");
        let table_id = row.uuid(1)?.expect("table_id is not null");
        let index_id = row.uuid(2)?.expect("index_id is not null");
        let data_file_id = row.uuid(3)?.expect("data_file_id is not null");
        let relative_path = row.utf8_owned(4)?.expect("relative_path is not null");
        Ok(Self {
            index_file_id,
            table_id,
            index_id,
            data_file_id,
            relative_path,
        })
    }
}

pub(crate) struct InlineIndexRecord {
    pub(crate) index_id: Uuid,
    pub(crate) index_data: Vec<u8>,
}

impl InlineIndexRecord {
    pub(crate) fn to_sql(&self, database: CatalogDatabase) -> String {
        format!(
            "({}, {})",
            database.sql_uuid_literal(&self.index_id),
            database.sql_binary_literal(&self.index_data)
        )
    }

    pub(crate) fn catalog_schema() -> CatalogSchema {
        CatalogSchema::new(vec![
            Column::new("index_id", CatalogDataType::Uuid, false),
            Column::new("index_data", CatalogDataType::Binary, false),
        ])
    }

    pub(crate) fn from_row(mut row: Row) -> ILResult<Self> {
        let index_id = row.uuid(0)?.expect("index_id is not null");
        let index_data = row.binary_owned(1)?.expect("index_data is not null");
        Ok(Self {
            index_id,
            index_data,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_validity() {
        let mut row_validity = RowValidity::new(5);
        row_validity.set(2, false);
        let bool_vec = row_validity.iter().collect::<Vec<bool>>();
        assert_eq!(bool_vec, vec![true, true, false, true, true])
    }
}
