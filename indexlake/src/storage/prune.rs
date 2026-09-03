//! Row-group pruning from parquet footer statistics, judged by DataFusion's
//! pruning machinery.
//!
//! Before reading a data file, its footer is decoded once and handed to a
//! DataFusion [`PruningPredicate`] built from the scan filters. Row groups
//! that cannot contain a matching row are skipped via the reader's row
//! selection; a file whose row groups are all skipped is dropped from the
//! scan entirely.
//!
//! Pruning is deliberately conservative: on any conversion or evaluation
//! error (or missing statistic) the file (or row group) is kept, and the
//! in-memory row filter remains the authoritative filtering stage regardless
//! of what is pruned here.

use std::sync::Arc;

use arrow::array::{ArrayRef, BooleanArray, UInt64Array};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::DFSchema;
use datafusion_common::pruning::PruningStatistics;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_physical_expr::create_physical_expr;
use datafusion_pruning::PruningPredicateBuilder;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader, RowGroupMetaData};
use parquet::schema::types::SchemaDescriptor;

use crate::expr::Expr as ILExpr;
use crate::expr::datafusion::indexlake_expr_to_datafusion_expr;
use crate::storage::InputFile;
use crate::{ILError, ILResult};

/// Outcome of statistics-based pruning for one data file.
pub(crate) enum PruneOutcome {
    /// Every row group was pruned: the file cannot contain matching rows.
    Skip,
    /// Some row groups were pruned: read only the given row ranges.
    Partial(RowSelection),
}

/// A DataFusion pruning predicate prebuilt from scan filters, paired with the
/// table schema it was built against — the row-group statistics adapter needs
/// the same schema for column resolution.
///
/// The predicate only depends on the filters and the table schema, so it is
/// built once per scan and shared across all files; cloning is cheap (an
/// `Arc` and a schema reference).
#[derive(Clone)]
pub(crate) struct RowGroupPruner {
    predicate: Arc<datafusion_pruning::PruningPredicate>,
    table_schema: SchemaRef,
}

impl RowGroupPruner {
    fn prune(&self, metadata: &ParquetMetaData) -> Option<PruneOutcome> {
        let stats = RowGroupStats {
            arrow_schema: &self.table_schema,
            parquet_schema: metadata.file_metadata().schema_descr(),
            row_groups: metadata.row_groups(),
        };
        let keeps = self
            .predicate
            .prune(&stats)
            .inspect_err(|e| log::debug!("row group pruning evaluation failed: {e}"))
            .ok()?;

        let mut ranges = Vec::new();
        let mut any_pruned = false;
        let mut offset = 0usize;
        for (row_group, keep) in metadata.row_groups().iter().zip(keeps) {
            let num_rows = row_group.num_rows() as usize;
            if keep {
                ranges.push(offset..offset + num_rows);
            } else {
                any_pruned = true;
            }
            offset += num_rows;
        }
        if !any_pruned {
            return None;
        }
        if ranges.is_empty() {
            return Some(PruneOutcome::Skip);
        }
        Some(PruneOutcome::Partial(
            RowSelection::from_consecutive_ranges(ranges.into_iter(), offset),
        ))
    }
}

/// Convert the scan filters into a row-group pruner. Returns [`None`] —
/// meaning "do not prune" — when the filters cannot be handed to datafusion
/// (conversion failure, e.g. indexlake functions, or physical planning
/// failure, or empty filters). This is always safe: the in-memory row filter
/// remains authoritative.
pub(crate) fn build_row_group_pruner(
    filters: &[ILExpr],
    table_schema: &SchemaRef,
) -> Option<RowGroupPruner> {
    build_pruning_predicate(filters, table_schema).map(|predicate| RowGroupPruner {
        predicate,
        table_schema: table_schema.clone(),
    })
}

/// Decide which row groups of a file can be skipped based on footer statistics.
///
/// The caller supplies a [`RowGroupPruner`] prebuilt from the scan filters —
/// once per scan, shared across all files.
///
/// Returns [`None`] when no row group was pruned (no pruner, or pruning is
/// unsafe), so the file can be read with its validity selection only. The
/// caller must pass the file's `record_count` from the catalog: a mismatch
/// with the footer's row group total means the selection grids are misaligned
/// and intersecting them could resurrect deleted rows, so pruning is
/// abandoned.
pub(crate) fn prune_file_row_groups(
    metadata: &ParquetMetaData,
    pruner: Option<&RowGroupPruner>,
    record_count: usize,
) -> Option<PruneOutcome> {
    let footer_row_count: usize = metadata
        .row_groups()
        .iter()
        .map(|rg| rg.num_rows() as usize)
        .sum();
    if footer_row_count != record_count {
        return None;
    }
    let pruner = pruner?;
    pruner.prune(metadata)
}

/// Convert the scan filters into a DataFusion pruning predicate. Returns
/// [`None`] — meaning "do not prune" — when the filters cannot be handed to
/// datafusion (conversion failure, e.g. indexlake functions, or physical
/// planning failure, or empty filters). This is always safe: the in-memory
/// row filter remains authoritative.
fn build_pruning_predicate(
    filters: &[ILExpr],
    table_schema: &SchemaRef,
) -> Option<Arc<datafusion_pruning::PruningPredicate>> {
    let mut df_exprs = Vec::with_capacity(filters.len());
    for filter in filters {
        df_exprs.push(indexlake_expr_to_datafusion_expr(filter).ok()?);
    }
    let expr = df_exprs
        .into_iter()
        .reduce(|acc, expr| acc.and(expr))
        // An empty filter list yields no conjunct: nothing to prune with.
        ?;

    let df_schema = DFSchema::try_from(table_schema.clone()).ok()?;
    let physical_expr = create_physical_expr(
        &expr,
        &df_schema,
        &ExecutionProps::new(),
        &PhysicalPlanningContext::default(),
    )
    .ok()?;

    PruningPredicateBuilder::new()
        .with_file_schema(table_schema.clone())
        .try_build(physical_expr)
        .map(Arc::new)
        .ok()
}

/// Adapts parquet row-group metadata to DataFusion's [`PruningStatistics`]
/// over per-row-group min/max/null-count statistics, mirroring datafusion's
/// own (crate-private) row group pruner. Statistics conversion failures for a
/// single column yield `None`, which datafusion treats as "no information":
/// the row group is kept.
struct RowGroupStats<'a> {
    arrow_schema: &'a Schema,
    parquet_schema: &'a SchemaDescriptor,
    row_groups: &'a [RowGroupMetaData],
}

impl RowGroupStats<'_> {
    fn converter(&self, column: &datafusion_common::Column) -> Option<StatisticsConverter<'_>> {
        StatisticsConverter::try_new(&column.name, self.arrow_schema, self.parquet_schema).ok()
    }
}

impl PruningStatistics for RowGroupStats<'_> {
    fn min_values(&self, column: &datafusion_common::Column) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_mins(self.row_groups.iter())
            .ok()
    }

    fn max_values(&self, column: &datafusion_common::Column) -> Option<ArrayRef> {
        self.converter(column)?
            .row_group_maxes(self.row_groups.iter())
            .ok()
    }

    fn num_containers(&self) -> usize {
        self.row_groups.len()
    }

    fn null_counts(&self, column: &datafusion_common::Column) -> Option<ArrayRef> {
        let counts = self
            .converter(column)?
            .with_missing_null_counts_as_zero(false)
            .row_group_null_counts(self.row_groups.iter())
            .ok()?;
        Some(Arc::new(UInt64Array::from(counts)) as ArrayRef)
    }

    fn row_counts(&self) -> Option<ArrayRef> {
        let counts: UInt64Array = self
            .row_groups
            .iter()
            .map(|rg| Some(rg.num_rows() as u64))
            .collect();
        Some(Arc::new(counts) as ArrayRef)
    }

    fn contained(
        &self,
        _column: &datafusion_common::Column,
        _values: &std::collections::HashSet<datafusion_common::ScalarValue>,
    ) -> Option<BooleanArray> {
        // Value-set membership cannot be judged from min/max statistics alone.
        None
    }
}

/// Number of suffix bytes to prefetch for the footer: size/12 (datafusion's
/// default heuristic), at least 64KB, clamped to the file size. Single source
/// shared by the footer read and any other metadata read.
pub(crate) fn footer_size_hint(size: u64) -> usize {
    std::cmp::min(size as usize, std::cmp::max(size as usize / 12, 64 * 1024))
}

/// Read and decode the footer metadata of a parquet file through an already
/// opened input file, normally in a single suffix range read. The reader
/// builder reuses the same input file afterwards, so the footer is read
/// exactly once per file.
pub(crate) async fn read_footer_metadata(
    input_file: &mut dyn InputFile,
    relative_path: &str,
    size: u64,
) -> ILResult<ParquetMetaData> {
    if size < 12 {
        return Err(ILError::internal(format!(
            "Parquet file {relative_path} is too small to hold a footer"
        )));
    }
    let hint = footer_size_hint(size);
    let suffix = input_file.read((size - hint as u64)..size).await?;
    let n = suffix.len();
    // Trailer layout: [metadata_len: u32][PAR1]
    if n < 8 || &suffix[n - 4..] != b"PAR1" {
        return Err(ILError::internal(format!(
            "Parquet file {relative_path} has an invalid footer trailer"
        )));
    }
    let footer_len = u32::from_le_bytes(suffix[n - 8..n - 4].try_into().unwrap()) as u64;
    if footer_len + 8 > size {
        // Corrupted trailer: the claimed footer cannot fit in the file.
        return Err(ILError::internal(format!(
            "Parquet file {relative_path} has an invalid footer length"
        )));
    }
    let footer_len = footer_len as usize;
    let metadata = if footer_len <= n - 8 {
        suffix.slice(n - 8 - footer_len..n - 8)
    } else {
        // Footer is larger than the prefetched suffix: fetch it exactly.
        input_file
            .read((size - 8 - footer_len as u64)..(size - 8))
            .await?
    };
    ParquetMetaDataReader::decode_metadata(metadata.as_ref()).map_err(|e| {
        ILError::internal(format!(
            "Failed to decode parquet metadata of {relative_path}: {e}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::RowSelection;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::properties::WriterProperties;
    use parquet::file::statistics::Statistics;

    use super::*;
    use crate::expr::{col, lit};

    /// Write the given columns into an in-memory parquet file with the given
    /// rows per row group and return its footer metadata.
    fn write_test_parquet(
        schema: &SchemaRef,
        columns: Vec<ArrayRef>,
        rows_per_row_group: usize,
    ) -> ParquetMetaData {
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_row_group))
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::<u8>::new(), schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap()
    }

    fn int_column(values: &[i64]) -> (Field, ArrayRef) {
        (
            Field::new("grp", DataType::Int64, false),
            Arc::new(Int64Array::from(values.to_vec())) as ArrayRef,
        )
    }

    fn pruner(filters: &[ILExpr], schema: &SchemaRef) -> Option<RowGroupPruner> {
        build_row_group_pruner(filters, schema)
    }

    /// M1 defense: a footer whose row total disagrees with the catalog
    /// record count must abandon pruning instead of intersecting misaligned
    /// selections (which could resurrect deleted rows).
    #[test]
    fn prune_abandons_on_record_count_mismatch() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![col("grp").eq(lit(25i64))];
        assert!(prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30).is_some());
        assert!(prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 29).is_none());
        assert!(prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 31).is_none());
    }

    #[test]
    fn prune_keeps_only_row_groups_matching_predicate() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![col("grp").eq(lit(25i64))];
        match prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => {
                // Only the third row group (rows 20..30, min/max [20, 29])
                // can contain 25.
                assert_eq!(selection.row_count(), 10);
            }
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn prune_skips_file_when_no_row_group_matches() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![col("grp").eq(lit(500i64))];
        assert!(matches!(
            prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30),
            Some(PruneOutcome::Skip)
        ));
    }

    /// A fully deleted file has an empty validity selection; intersecting it
    /// with any row-group selection must stay empty.
    #[test]
    fn all_deleted_file_intersects_to_empty() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![col("grp").eq(lit(25i64))];
        let Some(PruneOutcome::Partial(row_group_selection)) =
            prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30)
        else {
            panic!("expected partial pruning");
        };
        let all_deleted = RowSelection::from_consecutive_ranges(std::iter::empty(), 30);
        assert_eq!(
            all_deleted.intersection(&row_group_selection).row_count(),
            0
        );
    }

    /// Deleted rows crossing a row-group boundary: validity keeps rows
    /// [0..5) and [15..25) of 30, the predicate keeps row group [20..30).
    /// The intersection must keep exactly rows [20..25).
    #[test]
    fn deleted_rows_across_row_groups_intersect_correctly() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![col("grp").eq(lit(25i64))];
        let Some(PruneOutcome::Partial(row_group_selection)) =
            prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30)
        else {
            panic!("expected partial pruning");
        };
        let validity = RowSelection::from_consecutive_ranges([0..5, 15..25].into_iter(), 30);
        let selection = validity.intersection(&row_group_selection);
        assert_eq!(selection.row_count(), 5);
    }

    /// Float predicates were never pruned by the old hand-written statistics
    /// matcher; datafusion's pruning machinery prunes them like any other
    /// ordered type.
    #[test]
    fn prune_supports_float_predicates() {
        let values: Vec<f64> = (0..30).map(|i| i as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Float64Array::from(values)) as ArrayRef],
            10,
        );
        let filters = vec![col("value").eq(lit(25.5f64))];
        match prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => {
                assert_eq!(selection.row_count(), 10);
            }
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn missing_null_counts_do_not_prune_is_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(2)])) as ArrayRef],
            10,
        );

        let mut metadata_builder = metadata.into_builder();
        let mut row_groups = metadata_builder.take_row_groups();
        let row_group = row_groups.pop().unwrap();
        let column = row_group.column(0);
        let statistics = match column.statistics().unwrap() {
            Statistics::Int64(stats) => Statistics::int64(
                stats.min_opt().copied(),
                stats.max_opt().copied(),
                stats.distinct_count(),
                None,
                false,
            ),
            other => panic!("expected Int64 statistics, got {other:?}"),
        };
        let column = column
            .clone()
            .into_builder()
            .set_statistics(statistics)
            .build()
            .unwrap();
        let row_group = row_group
            .into_builder()
            .set_column_metadata(vec![column])
            .build()
            .unwrap();
        row_groups.push(row_group);
        let metadata = metadata_builder.set_row_groups(row_groups).build();

        let filters = vec![col("value").is_null()];
        assert!(prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 3).is_none());
    }

    #[test]
    fn prune_supports_utf8_predicates() {
        let values: Vec<String> = (0..30).map(|i| format!("key_{i:03}")).collect();
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from(values)) as ArrayRef],
            10,
        );
        let filters = vec![col("name").eq(lit("key_025".to_string()))];
        match prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => {
                assert_eq!(selection.row_count(), 10);
            }
            _ => panic!("expected partial pruning"),
        }
    }

    /// Indexlake function expressions cannot be handed to datafusion; pruning
    /// must be abandoned (keeping every row group) instead of erroring.
    #[test]
    fn prune_abandons_unconvertible_filters() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let filters = vec![crate::expr::Expr::Function(crate::expr::Function {
            name: "version".to_string(),
            args: vec![],
            return_type: DataType::Int64,
        })];
        assert!(prune_file_row_groups(&metadata, pruner(&filters, &schema).as_ref(), 30).is_none());
    }

    /// Empty filters prune nothing.
    #[test]
    fn prune_does_nothing_without_filters() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        assert!(prune_file_row_groups(&metadata, None, 30).is_none());
    }

    /// `contained` reports no value-set knowledge: min/max statistics alone
    /// cannot decide membership.
    #[test]
    fn row_group_stats_reports_no_contained_knowledge() {
        let schema = Arc::new(Schema::new(vec![
            int_column(&(0..30).collect::<Vec<_>>()).0,
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![int_column(&(0..30).collect::<Vec<_>>()).1],
            10,
        );
        let stats = RowGroupStats {
            arrow_schema: &schema,
            parquet_schema: metadata.file_metadata().schema_descr(),
            row_groups: metadata.row_groups(),
        };
        let mut values = HashSet::new();
        values.insert(datafusion_common::ScalarValue::Int64(Some(1)));
        assert!(PruningStatistics::contained(&stats, &"grp".into(), &values).is_none());
    }
}
