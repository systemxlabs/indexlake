//! Best-effort row-group pruning from parquet footer statistics.
//!
//! Before reading a data file, its footer is decoded once and every row
//! group's per-column min/max statistics are checked against the filter's
//! comparison predicates. Row groups that cannot contain a matching row are
//! skipped via the reader's row selection; a file whose row groups are all
//! skipped is dropped from the scan entirely.
//!
//! Pruning is deliberately conservative: on any error or missing statistic
//! the file (or row group) is kept, and the in-memory row filter remains
//! the authoritative filtering stage regardless of what is pruned here.

use arrow::datatypes::{DataType, SchemaRef, TimeUnit};
use parquet::arrow::arrow_reader::RowSelection;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader, RowGroupMetaData};
use parquet::file::statistics::Statistics;

use crate::catalog::Scalar;
use crate::expr::{BinaryOp, Expr, split_conjunction_filters};
use crate::storage::InputFile;
use crate::{ILError, ILResult};

/// A comparison predicate that can be evaluated against column statistics:
/// `column <op> constant`.
pub(crate) struct StatsPredicate {
    /// Index of the column in the table schema. Only valid for flat schemas,
    /// where the parquet leaf column order matches the arrow field order.
    column_index: usize,
    /// Name of the column, checked against the parquet column name before
    /// any statistics are trusted.
    column_name: String,
    op: BinaryOp,
    value: StatsValue,
}

/// The constant side of a [`StatsPredicate`], restricted to types whose
/// statistics ordering is trustworthy.
///
/// Float pruning is never enabled: parquet statistics for float columns are
/// historically dirty (NaN handling changed across format versions), so their
/// min/max ordering cannot be trusted. Decimal and date columns are also
/// conservatively excluded. "Unsupported means no pruning" — correctness is
/// never at stake.
#[derive(Debug, Clone, PartialEq)]
enum StatsValue {
    /// INT64 statistics (int64 and timestamp columns).
    Int(i64),
    /// BYTE_ARRAY statistics (utf8 columns), compared lexicographically.
    Str(String),
}

/// Outcome of statistics-based pruning for one data file.
pub(crate) enum PruneOutcome {
    /// Every row group was pruned: the file cannot contain matching rows.
    Skip,
    /// Some row groups were pruned: read only the given row ranges.
    Partial(RowSelection),
}

/// Flatten AND-conjunctions and keep only the leaves that can be evaluated
/// against parquet column statistics (`column <op> constant` over an int64,
/// timestamp, or utf8 column).
///
/// Unsupported leaves are dropped rather than disabling pruning: under AND
/// semantics each kept predicate can independently prune rows, so dropping a
/// leaf is always safe. Returns an empty vector when nothing is usable, in
/// which case pruning should not run at all.
pub(crate) fn extract_stats_predicates(
    filters: &[Expr],
    schema: &SchemaRef,
) -> Vec<StatsPredicate> {
    // Nested types would break the arrow-field-to-parquet-leaf index mapping.
    if schema.fields().iter().any(|f| f.data_type().is_nested()) {
        return Vec::new();
    }

    let mut predicates = Vec::new();
    for filter in filters {
        for leaf in split_conjunction_filters(vec![filter.clone()]) {
            if let Some(predicate) = extract_leaf_predicate(&leaf, schema) {
                predicates.push(predicate);
            }
        }
    }
    predicates
}

fn extract_leaf_predicate(leaf: &Expr, schema: &SchemaRef) -> Option<StatsPredicate> {
    let Expr::BinaryExpr(binary) = leaf else {
        return None;
    };
    let op = binary.op;
    if !matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    ) {
        return None;
    }
    // Normalize to (column, literal), mirroring the operator when the literal
    // is on the left-hand side.
    let (column_name, op, value) = match (&*binary.left, &*binary.right) {
        (Expr::Column(name), Expr::Literal(literal)) => (name, op, &literal.value),
        (Expr::Literal(literal), Expr::Column(name)) => (name, mirror_op(op), &literal.value),
        _ => return None,
    };
    let column_index = schema.index_of(column_name).ok()?;
    let data_type = schema.field(column_index).data_type();
    let column_name = column_name.clone();
    let value = match (data_type, value) {
        (DataType::Int64, Scalar::Int64(Some(v))) => StatsValue::Int(*v),
        (DataType::Timestamp(TimeUnit::Second, _), Scalar::TimestampSecond(Some(v), _)) => {
            StatsValue::Int(*v)
        }
        (
            DataType::Timestamp(TimeUnit::Millisecond, _),
            Scalar::TimestampMillisecond(Some(v), _),
        ) => StatsValue::Int(*v),
        (
            DataType::Timestamp(TimeUnit::Microsecond, _),
            Scalar::TimestampMicrosecond(Some(v), _),
        ) => StatsValue::Int(*v),
        (DataType::Timestamp(TimeUnit::Nanosecond, _), Scalar::TimestampNanosecond(Some(v), _)) => {
            StatsValue::Int(*v)
        }
        (DataType::Utf8, Scalar::Utf8(Some(v))) => StatsValue::Str(v.clone()),
        _ => return None,
    };
    Some(StatsPredicate {
        column_index,
        column_name,
        op,
        value,
    })
}

fn mirror_op(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        other => other,
    }
}

/// Decide which row groups of a file can be skipped based on footer statistics.
///
/// Returns [`None`] when no row group was pruned (or pruning is unsafe), so
/// the file can be read with its validity selection only. The caller must
/// pass the file's `record_count` from the catalog: a mismatch with the
/// footer's row group total means the selection grids are misaligned and
/// intersecting them could resurrect deleted rows, so pruning is abandoned.
pub(crate) fn prune_file_row_groups(
    metadata: &ParquetMetaData,
    predicates: &[StatsPredicate],
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
    prune_row_groups(metadata, predicates)
}

/// Decide which row groups of a file can be skipped. Returns [`None`] when no
/// row group was pruned, so the file can be read without a modified selection.
fn prune_row_groups(
    metadata: &ParquetMetaData,
    predicates: &[StatsPredicate],
) -> Option<PruneOutcome> {
    let mut ranges = Vec::new();
    let mut any_pruned = false;
    let mut offset = 0usize;
    for row_group in metadata.row_groups() {
        let num_rows = row_group.num_rows() as usize;
        let keep = predicates
            .iter()
            .all(|p| predicate_could_match_in_row_group(row_group, p, num_rows));
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

/// Whether any non-null value in the row group could satisfy the predicate,
/// judging only by the column chunk's min/max statistics. Returns false only
/// when it is guaranteed that no row satisfies it.
fn predicate_could_match_in_row_group(
    row_group: &RowGroupMetaData,
    predicate: &StatsPredicate,
    num_rows: usize,
) -> bool {
    if predicate.column_index >= row_group.num_columns() {
        return true;
    }
    let column = row_group.column(predicate.column_index);
    // Guard against files whose column order drifted from the table schema:
    // only trust statistics of the column with the expected name.
    if column.column_descr().name() != predicate.column_name {
        return true;
    }
    let Some(statistics) = column.statistics() else {
        return true;
    };
    // Nulls never satisfy a comparison, so an all-null chunk matches nothing.
    if statistics.null_count_opt() == Some(num_rows as u64) {
        return false;
    }
    match (&predicate.value, statistics) {
        (StatsValue::Int(v), Statistics::Int64(stats)) => {
            match (stats.min_opt(), stats.max_opt()) {
                (Some(min), Some(max)) => int_range_could_match(predicate.op, *min, *max, *v),
                _ => true,
            }
        }
        (StatsValue::Str(v), Statistics::ByteArray(stats)) => {
            match (stats.min_opt(), stats.max_opt()) {
                (Some(min), Some(max)) => {
                    str_range_could_match(predicate.op, min.data(), max.data(), v.as_bytes())
                }
                _ => true,
            }
        }
        _ => true,
    }
}

fn int_range_could_match(op: BinaryOp, min: i64, max: i64, value: i64) -> bool {
    match op {
        BinaryOp::Eq => value >= min && value <= max,
        BinaryOp::NotEq => !(min == max && min == value),
        BinaryOp::Lt => min < value,
        BinaryOp::LtEq => min <= value,
        BinaryOp::Gt => max > value,
        BinaryOp::GtEq => max >= value,
        _ => true,
    }
}

fn str_range_could_match(op: BinaryOp, min: &[u8], max: &[u8], value: &[u8]) -> bool {
    match op {
        BinaryOp::Eq => value >= min && value <= max,
        BinaryOp::NotEq => !(min == max && min == value),
        BinaryOp::Lt => min < value,
        BinaryOp::LtEq => min <= value,
        BinaryOp::Gt => max > value,
        BinaryOp::GtEq => max >= value,
        _ => true,
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
    use std::io::Cursor;
    use std::sync::Arc;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::arrow_reader::RowSelection;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::metadata::ParquetMetaDataReader;
    use parquet::file::properties::WriterProperties;

    use super::*;
    use crate::expr::{col, lit};

    /// Write one int64 column ("grp") into an in-memory parquet file with the
    /// given rows per row group and decode its footer metadata.
    fn write_test_parquet(values: &[i64], rows_per_row_group: usize) -> ParquetMetaData {
        let schema = Arc::new(Schema::new(vec![Field::new("grp", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int64Array::from(values.to_vec()))],
        )
        .unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_row_group))
            .build();
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = ArrowWriter::try_new(&mut cursor, schema, Some(props)).unwrap();
            writer.write(&batch).unwrap();
            writer.close().unwrap();
        }
        ParquetMetaDataReader::decode_metadata(cursor.into_inner().as_slice()).unwrap()
    }

    fn eq_predicates(value: i64) -> Vec<StatsPredicate> {
        let schema = Arc::new(Schema::new(vec![Field::new("grp", DataType::Int64, false)]));
        extract_stats_predicates(&[col("grp").eq(lit(value))], &schema)
    }

    /// M1 defense: a footer whose row total disagrees with the catalog
    /// record count must abandon pruning instead of intersecting misaligned
    /// selections (which could resurrect deleted rows).
    #[test]
    fn prune_abandons_on_record_count_mismatch() {
        let metadata = write_test_parquet(&(0..30).collect::<Vec<_>>(), 10);
        let predicates = eq_predicates(25);
        assert!(prune_file_row_groups(&metadata, &predicates, 30).is_some());
        assert!(prune_file_row_groups(&metadata, &predicates, 29).is_none());
        assert!(prune_file_row_groups(&metadata, &predicates, 31).is_none());
    }

    #[test]
    fn prune_keeps_only_row_groups_matching_predicate() {
        let metadata = write_test_parquet(&(0..30).collect::<Vec<_>>(), 10);
        let predicates = eq_predicates(25);
        match prune_file_row_groups(&metadata, &predicates, 30) {
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
        let metadata = write_test_parquet(&(0..30).collect::<Vec<_>>(), 10);
        let predicates = eq_predicates(500);
        assert!(matches!(
            prune_file_row_groups(&metadata, &predicates, 30),
            Some(PruneOutcome::Skip)
        ));
    }

    /// A fully deleted file has an empty validity selection; intersecting it
    /// with any row-group selection must stay empty.
    #[test]
    fn all_deleted_file_intersects_to_empty() {
        let metadata = write_test_parquet(&(0..30).collect::<Vec<_>>(), 10);
        let predicates = eq_predicates(25);
        let Some(PruneOutcome::Partial(row_group_selection)) =
            prune_file_row_groups(&metadata, &predicates, 30)
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
        let metadata = write_test_parquet(&(0..30).collect::<Vec<_>>(), 10);
        let predicates = eq_predicates(25);
        let Some(PruneOutcome::Partial(row_group_selection)) =
            prune_file_row_groups(&metadata, &predicates, 30)
        else {
            panic!("expected partial pruning");
        };
        let validity = RowSelection::from_consecutive_ranges([0..5, 15..25].into_iter(), 30);
        let selection = validity.intersection(&row_group_selection);
        assert_eq!(selection.row_count(), 5);
    }

    /// Float (and other non-whitelisted) columns must be conservatively
    /// dropped from statistics pruning; only int64 and utf8 survive.
    #[test]
    fn extract_stats_predicates_drops_unsupported_types() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Float64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        let filters = vec![
            col("a").eq(lit(1i64)),
            col("b").eq(lit(1.5f64)),
            col("c").eq(lit("x".to_string())),
        ];
        let predicates = extract_stats_predicates(&filters, &schema);
        assert_eq!(predicates.len(), 2);
    }
}
