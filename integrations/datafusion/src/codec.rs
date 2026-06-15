use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::DataFusionError;
use datafusion_execution::TaskContext;
use datafusion_expr::dml::InsertOp;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_proto::logical_plan::DefaultLogicalExtensionCodec;
use datafusion_proto::logical_plan::from_proto::parse_exprs;
use datafusion_proto::logical_plan::to_proto::serialize_exprs;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_proto::protobuf::Schema as ProtoSchema;
use indexlake::catalog::{DataFileRecord, RowValidity};
use indexlake::index::SearchQuery;
use indexlake::storage::DataFileFormat;
use indexlake::table::TableUpdate;
use prost::Message;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::index_lake_physical_plan_node::IndexLakePhysicalPlanType;
use crate::{
    DataFile, ExprColumnAssignment, IndexLakeDeleteExec, IndexLakeDeleteExecNode,
    IndexLakeExprNode, IndexLakeInsertExec, IndexLakeInsertExecNode, IndexLakePhysicalPlanNode,
    IndexLakeScanExec, IndexLakeScanExecNode, IndexLakeSearchExec, IndexLakeSearchExecNode,
    IndexLakeUpdateExec, IndexLakeUpdateExecNode, LazyTable, TableScanPartition,
    TableScanPartitionAuto, TableScanPartitionProvided, table_scan_partition,
};

#[derive(Debug)]
pub struct IndexLakePhysicalCodec {
    client: Arc<indexlake::Client>,
}

impl IndexLakePhysicalCodec {
    pub fn new(client: Arc<indexlake::Client>) -> Self {
        Self { client }
    }
}

impl PhysicalExtensionCodec for IndexLakePhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let indexlake_node = IndexLakePhysicalPlanNode::decode(buf).map_err(|e| {
            DataFusionError::Internal(format!(
                "Failed to decode indexlake physical plan node: {e:?}"
            ))
        })?;
        let indexlake_plan = indexlake_node.index_lake_physical_plan_type.ok_or_else(|| {
            DataFusionError::Internal(
                "Failed to decode indexlake physical plan node due to physical plan type is none".to_string()
            )
        })?;

        match indexlake_plan {
            IndexLakePhysicalPlanType::Scan(node) => {
                let schema = parse_schema(node.schema)?;

                let projection = parse_projection(node.projection.as_ref());
                let filters = parse_exprs(&node.filters, ctx, &DefaultLogicalExtensionCodec {})?;

                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                let scan_partitions = parse_scan_partitions(&node.partitions)?;
                let partition_row_counts = node
                    .partition_row_counts
                    .iter()
                    .map(|value| *value as usize)
                    .collect::<Vec<_>>();
                let partition_row_counts = Arc::new(partition_row_counts);

                Ok(Arc::new(IndexLakeScanExec::try_new(
                    lazy_table,
                    schema,
                    scan_partitions,
                    partition_row_counts,
                    projection,
                    filters,
                    node.batch_size as usize,
                    node.limit.map(|l| l as usize),
                )?))
            }
            IndexLakePhysicalPlanType::Insert(node) => {
                if inputs.len() != 1 {
                    return Err(DataFusionError::Internal(format!(
                        "IndexLakeInsertExec requires exactly one input, got {}",
                        inputs.len()
                    )));
                }
                let input = inputs[0].clone();

                let insert_op = parse_insert_op(node.insert_op)?;

                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                Ok(Arc::new(IndexLakeInsertExec::try_new(
                    lazy_table,
                    input,
                    insert_op,
                    node.bypass_insert_threshold as usize,
                )?))
            }
            IndexLakePhysicalPlanType::Search(node) => {
                let schema = parse_schema(node.schema)?;
                let projection = parse_projection(node.projection.as_ref());
                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                // Search query is decoded with basic info; type-specific params
                // require the SearchQueryCodec registered on the caller's side.
                let query = Arc::new(GenericSearchQuery {
                    index_kind: node.index_kind.clone(),
                    limit: node.limit.map(|l| l as usize),
                });

                Ok(Arc::new(IndexLakeSearchExec::try_new(
                    lazy_table,
                    schema,
                    query,
                    node.dynamic_fields.clone(),
                    projection,
                )?))
            }
            IndexLakePhysicalPlanType::Update(node) => {
                let condition: indexlake::expr::Expr = serde_json::from_str(
                    &node
                        .condition
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "Missing condition in update node".to_string(),
                            )
                        })?
                        .json,
                )
                .map_err(|e| {
                    DataFusionError::Internal(format!(
                        "Failed to deserialize update condition: {e}"
                    ))
                })?;

                let mut set_map = HashMap::new();
                for assignment in &node.assignments {
                    let value: indexlake::expr::Expr = serde_json::from_str(
                        &assignment
                            .value
                            .as_ref()
                            .ok_or_else(|| {
                                DataFusionError::Internal(
                                    "Missing value in update assignment".to_string(),
                                )
                            })?
                            .json,
                    )
                    .map_err(|e| {
                        DataFusionError::Internal(format!(
                            "Failed to deserialize update assignment: {e}"
                        ))
                    })?;
                    set_map.insert(assignment.column.clone(), value);
                }

                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                let update = TableUpdate { set_map, condition };
                Ok(Arc::new(IndexLakeUpdateExec::try_new(lazy_table, update)?))
            }
            IndexLakePhysicalPlanType::Delete(node) => {
                let condition: indexlake::expr::Expr = serde_json::from_str(
                    &node
                        .condition
                        .ok_or_else(|| {
                            DataFusionError::Internal(
                                "Missing condition in delete node".to_string(),
                            )
                        })?
                        .json,
                )
                .map_err(|e| {
                    DataFusionError::Internal(format!(
                        "Failed to deserialize delete condition: {e}"
                    ))
                })?;

                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                Ok(Arc::new(IndexLakeDeleteExec::try_new(
                    lazy_table, condition,
                )?))
            }
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
    ) -> Result<(), DataFusionError> {
        if let Some(exec) = node.as_any().downcast_ref::<IndexLakeScanExec>() {
            let projection = serialize_projection(exec.projection.as_ref());

            let filters = serialize_exprs(&exec.filters, &DefaultLogicalExtensionCodec {})?;

            let schema = serialize_schema(&exec.output_schema)?;

            let partitions = serialize_scan_partitions(exec.scan_partitions());
            let partition_row_counts = exec
                .partition_row_counts()
                .iter()
                .map(|value| *value as u64)
                .collect();

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Scan(
                    IndexLakeScanExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        partition_count: exec.partition_count as u32,
                        partitions,
                        projection,
                        filters,
                        batch_size: exec.batch_size as u32,
                        limit: exec.limit.map(|l| l as u32),
                        schema: Some(schema),
                        partition_row_counts,
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to encode indexlake scan execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<IndexLakeInsertExec>() {
            let insert_op = serialize_insert_op(exec.insert_op);

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Insert(
                    IndexLakeInsertExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        insert_op,
                        bypass_insert_threshold: exec.bypass_insert_threshold as u32,
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to encode indexlake insert execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<IndexLakeSearchExec>() {
            let schema = serialize_schema(&exec.output_schema)?;
            let projection = serialize_projection(exec.projection.as_ref());

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Search(
                    IndexLakeSearchExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        index_kind: exec.query.index_kind().to_string(),
                        limit: exec.query.limit().map(|l| l as u32),
                        dynamic_fields: exec.dynamic_fields.clone(),
                        projection,
                        schema: Some(schema),
                        query_data: exec.query.encode(),
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to encode indexlake search execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<IndexLakeUpdateExec>() {
            let condition_json = serde_json::to_string(&exec.update.condition).map_err(|e| {
                DataFusionError::Internal(format!("Failed to serialize update condition: {e}"))
            })?;

            let assignments = exec
                .update
                .set_map
                .iter()
                .map(|(col, expr)| {
                    let value_json = serde_json::to_string(expr).map_err(|e| {
                        DataFusionError::Internal(format!("Failed to serialize update value: {e}"))
                    })?;
                    Ok(ExprColumnAssignment {
                        column: col.clone(),
                        value: Some(IndexLakeExprNode { json: value_json }),
                    })
                })
                .collect::<Result<Vec<_>, DataFusionError>>()?;

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Update(
                    IndexLakeUpdateExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        condition: Some(IndexLakeExprNode {
                            json: condition_json,
                        }),
                        assignments,
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to encode indexlake update execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else if let Some(exec) = node.as_any().downcast_ref::<IndexLakeDeleteExec>() {
            let condition_json = serde_json::to_string(&exec.condition).map_err(|e| {
                DataFusionError::Internal(format!("Failed to serialize delete condition: {e}"))
            })?;

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Delete(
                    IndexLakeDeleteExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        condition: Some(IndexLakeExprNode {
                            json: condition_json,
                        }),
                    },
                )),
            };

            proto.encode(buf).map_err(|e| {
                DataFusionError::Internal(format!(
                    "Failed to encode indexlake delete execution plan: {e:?}"
                ))
            })?;

            Ok(())
        } else {
            Err(DataFusionError::NotImplemented(format!(
                "IndexLakePhysicalCodec does not support encoding {}",
                node.name()
            )))
        }
    }
}

fn serialize_schema(schema: &SchemaRef) -> Result<ProtoSchema, DataFusionError> {
    let proto: ProtoSchema = schema
        .as_ref()
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("Failed to serialize schema: {e:?}")))?;
    Ok(proto)
}

fn parse_schema(proto: Option<ProtoSchema>) -> Result<SchemaRef, DataFusionError> {
    let proto =
        proto.ok_or_else(|| DataFusionError::Internal("Missing schema in protobuf".to_string()))?;
    let schema: arrow::datatypes::Schema = (&proto)
        .try_into()
        .map_err(|e| DataFusionError::Internal(format!("Failed to parse schema: {e:?}")))?;
    Ok(Arc::new(schema))
}

fn serialize_projection(projection: Option<&Vec<usize>>) -> Option<crate::protobuf::Projection> {
    projection.map(|p| crate::protobuf::Projection {
        projection: p.iter().map(|n| *n as u32).collect(),
    })
}

fn parse_projection(projection: Option<&crate::protobuf::Projection>) -> Option<Vec<usize>> {
    projection.map(|p| p.projection.iter().map(|n| *n as usize).collect())
}

fn serialize_insert_op(insert_op: InsertOp) -> i32 {
    let proto = match insert_op {
        InsertOp::Append => datafusion_proto::protobuf::InsertOp::Append,
        InsertOp::Overwrite => datafusion_proto::protobuf::InsertOp::Overwrite,
        InsertOp::Replace => datafusion_proto::protobuf::InsertOp::Replace,
    };
    proto.into()
}

fn parse_insert_op(insert_op: i32) -> Result<InsertOp, DataFusionError> {
    let proto = datafusion_proto::protobuf::InsertOp::try_from(insert_op)
        .map_err(|e| DataFusionError::Internal(format!("Failed to parse insert op: {e:?}")))?;
    match proto {
        datafusion_proto::protobuf::InsertOp::Append => Ok(InsertOp::Append),
        datafusion_proto::protobuf::InsertOp::Overwrite => Ok(InsertOp::Overwrite),
        datafusion_proto::protobuf::InsertOp::Replace => Ok(InsertOp::Replace),
    }
}

fn serialize_scan_partitions(
    partitions: &Arc<Vec<indexlake::table::TableScanPartition>>,
) -> Vec<TableScanPartition> {
    partitions
        .iter()
        .map(|partition| match partition {
            indexlake::table::TableScanPartition::Auto {
                partition_idx,
                partition_count,
            } => TableScanPartition {
                partition_type: Some(table_scan_partition::PartitionType::Auto(
                    TableScanPartitionAuto {
                        partition_idx: *partition_idx as u32,
                        partition_count: *partition_count as u32,
                    },
                )),
            },
            indexlake::table::TableScanPartition::Provided {
                contains_inline_rows,
                data_file_records,
            } => TableScanPartition {
                partition_type: Some(table_scan_partition::PartitionType::Provided(
                    TableScanPartitionProvided {
                        contains_inline_rows: *contains_inline_rows,
                        data_file_records: data_file_records
                            .iter()
                            .map(serialize_data_file_record)
                            .collect(),
                    },
                )),
            },
        })
        .collect()
}

fn parse_scan_partitions(
    proto_partitions: &[TableScanPartition],
) -> Result<Arc<Vec<indexlake::table::TableScanPartition>>, DataFusionError> {
    if proto_partitions.is_empty() {
        return Err(DataFusionError::Internal(
            "Missing scan partitions in indexlake scan exec node".to_string(),
        ));
    }

    let mut partitions = Vec::with_capacity(proto_partitions.len());
    for (idx, partition) in proto_partitions.iter().enumerate() {
        let partition = match &partition.partition_type {
            Some(table_scan_partition::PartitionType::Auto(auto)) => {
                indexlake::table::TableScanPartition::Auto {
                    partition_idx: auto.partition_idx as usize,
                    partition_count: auto.partition_count as usize,
                }
            }
            Some(table_scan_partition::PartitionType::Provided(provided)) => {
                let mut records = Vec::with_capacity(provided.data_file_records.len());
                for record in &provided.data_file_records {
                    records.push(parse_data_file_record(record)?);
                }
                indexlake::table::TableScanPartition::Provided {
                    contains_inline_rows: provided.contains_inline_rows,
                    data_file_records: records,
                }
            }
            None => {
                return Err(DataFusionError::Internal(format!(
                    "Missing partition type for scan partition {idx}"
                )));
            }
        };
        partitions.push(partition);
    }
    Ok(Arc::new(partitions))
}

fn serialize_data_file_record(record: &DataFileRecord) -> DataFile {
    DataFile {
        data_file_id: record.data_file_id.as_bytes().to_vec(),
        table_id: record.table_id.as_bytes().to_vec(),
        format: serialize_data_file_format(record.format),
        relative_path: record.relative_path.clone(),
        size: record.size,
        record_count: record.record_count,
        validity: record.validity.bytes().to_vec(),
        valid_record_count: record.valid_record_count,
    }
}

fn parse_data_file_record(proto_data_file: &DataFile) -> Result<DataFileRecord, DataFusionError> {
    Ok(DataFileRecord {
        data_file_id: Uuid::from_slice(&proto_data_file.data_file_id).map_err(|e| {
            DataFusionError::Internal(format!("Failed to parse data file id: {e:?}"))
        })?,
        table_id: Uuid::from_slice(&proto_data_file.table_id)
            .map_err(|e| DataFusionError::Internal(format!("Failed to parse table id: {e:?}")))?,
        format: parse_data_file_format(proto_data_file.format)?,
        relative_path: proto_data_file.relative_path.clone(),
        size: proto_data_file.size,
        record_count: proto_data_file.record_count,
        valid_record_count: proto_data_file.valid_record_count,
        validity: RowValidity::from(
            proto_data_file.validity.clone(),
            proto_data_file.record_count as usize,
        ),
    })
}

fn serialize_data_file_format(format: DataFileFormat) -> i32 {
    let proto_format = match format {
        DataFileFormat::ParquetV1 => crate::protobuf::DataFileFormat::ParquetV1,
        DataFileFormat::ParquetV2 => crate::protobuf::DataFileFormat::ParquetV2,
    };
    proto_format.into()
}

fn parse_data_file_format(format: i32) -> Result<DataFileFormat, DataFusionError> {
    let proto_format = crate::protobuf::DataFileFormat::try_from(format).map_err(|e| {
        DataFusionError::Internal(format!("Failed to parse data file format: {e:?}"))
    })?;
    match proto_format {
        crate::protobuf::DataFileFormat::ParquetV1 => Ok(DataFileFormat::ParquetV1),
        crate::protobuf::DataFileFormat::ParquetV2 => Ok(DataFileFormat::ParquetV2),
    }
}

/// A basic search query used for protobuf deserialization.
/// Stores the index kind and limit; type-specific query params
/// are not preserved across serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericSearchQuery {
    pub index_kind: String,
    pub limit: Option<usize>,
}

impl SearchQuery for GenericSearchQuery {
    fn index_kind(&self) -> &str {
        &self.index_kind
    }

    fn limit(&self) -> Option<usize> {
        self.limit
    }

    fn encode(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }
}
