use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::DataFusionError;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::logical_plan::DefaultLogicalExtensionCodec;
use datafusion_proto::logical_plan::from_proto::parse_exprs;
use datafusion_proto::logical_plan::to_proto::serialize_exprs;
use datafusion_proto::physical_plan::PhysicalExtensionCodec;
use datafusion_proto::protobuf::Schema as ProtoSchema;
use indexlake::catalog::{DataFileRecord, RowValidity};
use indexlake::storage::DataFileFormat;
use prost::Message;
use uuid::Uuid;

use crate::index_lake_physical_plan_node::IndexLakePhysicalPlanType;
use crate::{
    DataFile, DataFiles, IndexLakeInsertExec, IndexLakeInsertExecNode, IndexLakePhysicalPlanNode,
    IndexLakeScanExec, IndexLakeScanExecNode, LazyTable,
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
        registry: &dyn FunctionRegistry,
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

                let data_files = parse_data_files(node.data_files)?;

                let projection = parse_projection(node.projection.as_ref());
                let filters =
                    parse_exprs(&node.filters, registry, &DefaultLogicalExtensionCodec {})?;

                let lazy_table =
                    LazyTable::new(self.client.clone(), node.namespace_name, node.table_name);

                Ok(Arc::new(IndexLakeScanExec::try_new(
                    lazy_table,
                    schema,
                    node.partition_count as usize,
                    data_files,
                    node.concurrency.map(|c| c as usize),
                    projection,
                    filters,
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

            let data_files = serialize_data_files(exec.data_files.as_ref())?;

            let schema = serialize_schema(&exec.output_schema)?;

            let proto = IndexLakePhysicalPlanNode {
                index_lake_physical_plan_type: Some(IndexLakePhysicalPlanType::Scan(
                    IndexLakeScanExecNode {
                        namespace_name: exec.lazy_table.namespace_name.clone(),
                        table_name: exec.lazy_table.table_name.clone(),
                        partition_count: exec.partition_count as u32,
                        data_files,
                        concurrency: exec.concurrency.map(|c| c as u32),
                        projection,
                        filters,
                        limit: exec.limit.map(|l| l as u32),
                        schema: Some(schema),
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
    let schema: datafusion::arrow::datatypes::Schema = (&proto)
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

fn serialize_data_files(
    records: Option<&Arc<Vec<DataFileRecord>>>,
) -> Result<Option<DataFiles>, DataFusionError> {
    match records {
        Some(records) => {
            let mut proto_data_files = vec![];
            for record in records.iter() {
                proto_data_files.push(DataFile {
                    data_file_id: record.data_file_id.as_bytes().to_vec(),
                    table_id: record.table_id.as_bytes().to_vec(),
                    format: serialize_data_file_format(record.format),
                    relative_path: record.relative_path.clone(),
                    size: record.size,
                    record_count: record.record_count,
                    validity: record.validity.bytes().to_vec(),
                });
            }
            Ok(Some(DataFiles {
                files: proto_data_files,
            }))
        }
        None => Ok(None),
    }
}

fn parse_data_files(
    proto_data_files: Option<DataFiles>,
) -> Result<Option<Arc<Vec<DataFileRecord>>>, DataFusionError> {
    match proto_data_files {
        Some(proto_data_files) => {
            let mut records = vec![];
            for proto_data_file in proto_data_files.files {
                records.push(DataFileRecord {
                    data_file_id: Uuid::from_slice(&proto_data_file.data_file_id).map_err(|e| {
                        DataFusionError::Internal(format!("Failed to parse data file id: {e:?}"))
                    })?,
                    table_id: Uuid::from_slice(&proto_data_file.table_id).map_err(|e| {
                        DataFusionError::Internal(format!("Failed to parse table id: {e:?}"))
                    })?,
                    format: parse_data_file_format(proto_data_file.format)?,
                    relative_path: proto_data_file.relative_path,
                    size: proto_data_file.size,
                    record_count: proto_data_file.record_count,
                    validity: RowValidity::from(
                        proto_data_file.validity,
                        proto_data_file.record_count as usize,
                    ),
                });
            }
            Ok(Some(Arc::new(records)))
        }
        None => Ok(None),
    }
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
