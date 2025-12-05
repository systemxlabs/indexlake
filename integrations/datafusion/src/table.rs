use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::stats::Precision;
use datafusion::common::{DFSchema, Statistics};
use datafusion::datasource::TableType;
use datafusion::error::DataFusionError;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::logical_expr::TableProviderFilterPushDown;
use datafusion::logical_expr::dml::InsertOp;
use datafusion::physical_expr::aggregate::AggregateExprBuilder;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::aggregates::{AggregateExec, AggregateMode, PhysicalGroupBy};
use datafusion::physical_plan::expressions::col;
use datafusion::prelude::Expr;
use indexlake::index::FilterSupport;
use indexlake::table::{Table, TableScanPartition};
use indexlake::utils::schema_without_row_id;
use log::warn;

use crate::{
    IndexLakeInsertExec, IndexLakeScanExec, datafusion_expr_to_indexlake_expr,
    indexlake_scalar_to_datafusion_scalar, make_count_schema,
};

#[derive(Debug)]
pub struct IndexLakeTable {
    table: Arc<Table>,
    scan_partitions: usize,
    column_defaults: HashMap<String, Expr>,
    hide_row_id: bool,
    scan_concurrency: Option<usize>,
    insert_partitions: Option<usize>,
}

impl IndexLakeTable {
    pub fn try_new(table: Arc<Table>) -> Result<Self, DataFusionError> {
        let mut column_defaults = HashMap::new();
        for field_record in table.field_records.iter() {
            if let Some(default_value) = &field_record.default_value {
                let scalar_value = indexlake_scalar_to_datafusion_scalar(default_value)?;
                column_defaults.insert(
                    field_record.field_name.clone(),
                    Expr::Literal(scalar_value, None),
                );
            }
        }
        Ok(Self {
            table,
            scan_partitions: 16,
            column_defaults,
            hide_row_id: false,
            scan_concurrency: None,
            insert_partitions: None,
        })
    }

    pub fn with_scan_partitions(mut self, scan_partitions: usize) -> Self {
        self.scan_partitions = scan_partitions;
        self
    }

    pub fn with_hide_row_id(mut self, hide_row_id: bool) -> Self {
        self.hide_row_id = hide_row_id;
        self
    }

    pub fn with_scan_concurrency(mut self, scan_concurrency: Option<usize>) -> Self {
        self.scan_concurrency = scan_concurrency;
        self
    }

    pub fn with_insert_partitions(mut self, insert_partitions: Option<usize>) -> Self {
        self.insert_partitions = insert_partitions;
        self
    }
}

#[async_trait::async_trait]
impl TableProvider for IndexLakeTable {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        if self.hide_row_id {
            Arc::new(schema_without_row_id(&self.table.output_schema))
        } else {
            self.table.output_schema.clone()
        }
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.column_defaults.get(column)
    }

    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let data_file_count = self
            .table
            .data_file_count()
            .await
            .map_err(|e| DataFusionError::Internal(e.to_string()))?;
        let data_files = if data_file_count > 1000 {
            None
        } else {
            let records = self
                .table
                .data_file_records()
                .await
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            Some(Arc::new(records))
        };

        let il_projection = if let Some(df_projection) = projection
            && self.hide_row_id
        {
            Some(df_projection.iter().map(|i| i + 1).collect::<Vec<_>>())
        } else {
            projection.cloned()
        };

        let exec = IndexLakeScanExec::try_new(
            self.table.clone(),
            self.scan_partitions,
            data_files,
            self.scan_concurrency,
            il_projection,
            filters.to_vec(),
            limit,
        )?;
        Ok(Arc::new(exec))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>, DataFusionError> {
        let df_schema = DFSchema::try_from(self.table.output_schema.clone())?;
        let mut supports = Vec::with_capacity(filters.len());
        for filter in filters {
            let Ok(il_expr) = datafusion_expr_to_indexlake_expr(filter, &df_schema) else {
                supports.push(TableProviderFilterPushDown::Unsupported);
                continue;
            };
            let support = self
                .table
                .supports_filter(il_expr.clone())
                .map_err(|e| DataFusionError::Internal(e.to_string()))?;
            match support {
                FilterSupport::Exact => supports.push(TableProviderFilterPushDown::Exact),
                FilterSupport::Inexact => supports.push(TableProviderFilterPushDown::Inexact),
                FilterSupport::Unsupported => {
                    supports.push(TableProviderFilterPushDown::Unsupported)
                }
            }
        }
        Ok(supports)
    }

    fn statistics(&self) -> Option<Statistics> {
        let row_count_result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                self.table
                    .count(TableScanPartition::single_partition())
                    .await
            })
        });
        match row_count_result {
            Ok(row_count) => Some(Statistics {
                num_rows: Precision::Exact(row_count),
                total_byte_size: Precision::Absent,
                column_statistics: Statistics::unknown_column(&self.table.output_schema),
            }),
            Err(e) => {
                warn!(
                    "[indexlake] Error getting indexlake table {}.{} row count: {:?}",
                    self.table.namespace_name, self.table.table_name, e
                );
                None
            }
        }
    }

    async fn insert_into(
        &self,
        _state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>, DataFusionError> {
        let insert_exec = IndexLakeInsertExec::try_new(
            self.table.clone(),
            input,
            insert_op,
            self.insert_partitions,
        )?;

        let count_schema = make_count_schema();
        let agg_expr =
            AggregateExprBuilder::new(sum_udaf(), vec![col("count", count_schema.as_ref())?])
                .schema(count_schema.clone())
                .alias("count")
                .build()?;
        let group_by = PhysicalGroupBy::new_single(vec![]);
        let agg_exec = AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(agg_expr)],
            vec![None],
            Arc::new(insert_exec),
            count_schema,
        )?;
        Ok(Arc::new(agg_exec))
    }
}
