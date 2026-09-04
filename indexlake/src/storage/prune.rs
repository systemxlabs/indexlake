//! Conservative row-group pruning from Parquet footer statistics.
//!
//! This module mirrors the important safety properties of DataFusion's
//! pruning predicate without depending on DataFusion's physical-expression
//! crates. It evaluates an expression over intervals (`min..=max`) and SQL
//! three-valued boolean possibilities. A row group is skipped only when the
//! predicate cannot possibly evaluate to `TRUE` for any row in that group.
//!
//! Missing, malformed, or unsupported statistics always produce an unknown
//! result and therefore keep the row group. The row-level filter remains the
//! authoritative filtering stage.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef};
use arrow::datatypes::{DataType, SchemaRef};
use parquet::arrow::arrow_reader::RowSelection;
use parquet::arrow::arrow_reader::statistics::StatisticsConverter;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::schema::types::SchemaDescriptor;

use crate::catalog::Scalar;
use crate::expr::{BinaryExpr, BinaryOp, Expr, Literal, visited_columns};
use crate::storage::InputFile;
use crate::{ILError, ILResult};

/// Outcome of statistics-based pruning for one data file.
pub(crate) enum PruneOutcome {
    /// Every row group was pruned: the file cannot contain matching rows.
    Skip,
    /// Some row groups were pruned: read only the given row ranges.
    Partial(RowSelection),
}

/// A scan-level pruner. The expression tree is built once and evaluated
/// against the statistics of each file's row groups.
#[derive(Clone)]
pub(crate) struct RowGroupPruner {
    filters: Arc<Vec<Expr>>,
    required_columns: Arc<Vec<String>>,
    table_schema: SchemaRef,
}

impl RowGroupPruner {
    fn prune(&self, metadata: &ParquetMetaData) -> Option<PruneOutcome> {
        let statistics = FileStatistics::new(
            &self.table_schema,
            &self.required_columns,
            metadata.file_metadata().schema_descr(),
            metadata,
        );

        let mut kept_ranges = Vec::with_capacity(metadata.num_row_groups());
        let mut any_pruned = false;
        let mut offset = 0usize;

        for row_group_idx in 0..metadata.num_row_groups() {
            let row_count = metadata.row_groups()[row_group_idx].num_rows() as usize;
            let possible = self
                .filters
                .iter()
                .map(|filter| possible_predicate(filter, &statistics, row_group_idx))
                .fold(BoolPossibility::always_true(), BoolPossibility::and);

            if possible.can_true {
                kept_ranges.push(offset..offset + row_count);
            } else {
                any_pruned = true;
            }
            offset += row_count;
        }

        if !any_pruned {
            return None;
        }
        if kept_ranges.is_empty() {
            return Some(PruneOutcome::Skip);
        }
        Some(PruneOutcome::Partial(
            RowSelection::from_consecutive_ranges(kept_ranges.into_iter(), offset),
        ))
    }
}

/// Build a pruner once per scan. Building the expression representation is
/// intentionally independent of any individual file footer.
pub(crate) fn build_row_group_pruner(
    filters: &[Expr],
    table_schema: &SchemaRef,
) -> Option<RowGroupPruner> {
    if filters.is_empty() {
        return None;
    }

    let mut required_columns = HashSet::new();
    for filter in filters {
        required_columns.extend(visited_columns(filter));
    }
    let mut required_columns = required_columns.into_iter().collect::<Vec<_>>();
    required_columns.sort_unstable();

    Some(RowGroupPruner {
        filters: Arc::new(filters.to_vec()),
        required_columns: Arc::new(required_columns),
        table_schema: table_schema.clone(),
    })
}

/// Decide which row groups can be skipped. The footer row count must match the
/// catalog count before selections are intersected with delete validity; if it
/// does not, pruning is abandoned to avoid mixing different row coordinate
/// systems.
pub(crate) fn prune_file_row_groups(
    metadata: &ParquetMetaData,
    pruner: Option<&RowGroupPruner>,
    record_count: usize,
) -> Option<PruneOutcome> {
    let footer_row_count: usize = metadata
        .row_groups()
        .iter()
        .map(|row_group| row_group.num_rows() as usize)
        .sum();
    if footer_row_count != record_count {
        return None;
    }
    pruner?.prune(metadata)
}

/// Statistics for all columns and row groups in one file. The arrays returned
/// by `StatisticsConverter` are materialized once per column, so evaluating a
/// predicate does not repeatedly decode Parquet statistics.
struct FileStatistics {
    columns: HashMap<String, Vec<ColumnStatistics>>,
}

#[derive(Clone)]
struct ColumnStatistics {
    min: Option<Scalar>,
    max: Option<Scalar>,
    null_count: Option<u64>,
    row_count: u64,
}

impl FileStatistics {
    fn new(
        schema: &SchemaRef,
        required_columns: &[String],
        parquet_schema: &SchemaDescriptor,
        metadata: &ParquetMetaData,
    ) -> Self {
        let row_counts = metadata
            .row_groups()
            .iter()
            .map(|row_group| row_group.num_rows() as u64)
            .collect::<Vec<_>>();
        let mut columns = HashMap::with_capacity(required_columns.len());

        for name in required_columns {
            let Ok(field) = schema.field_with_name(name) else {
                continue;
            };
            let Some(converter) = StatisticsConverter::try_new(name, schema, parquet_schema).ok()
            else {
                continue;
            };
            let Ok(mins) = converter.row_group_mins(metadata.row_groups().iter()) else {
                continue;
            };
            let Ok(maxes) = converter.row_group_maxes(metadata.row_groups().iter()) else {
                continue;
            };
            let Ok(null_counts) = converter
                .with_missing_null_counts_as_zero(false)
                .row_group_null_counts(metadata.row_groups().iter())
            else {
                continue;
            };

            let stats = (0..metadata.num_row_groups())
                .map(|idx| ColumnStatistics {
                    // Parquet writers exclude NaN from float min/max, while
                    // Arrow comparisons give NaN ordering semantics. Using
                    // those bounds here could skip a matching NaN row, so
                    // float columns deliberately retain only nullability
                    // information and fall back to row-level filtering.
                    min: if is_float_type(field.data_type()) {
                        None
                    } else {
                        scalar_at(&mins, idx)
                    },
                    max: if is_float_type(field.data_type()) {
                        None
                    } else {
                        scalar_at(&maxes, idx)
                    },
                    null_count: null_counts.is_valid(idx).then(|| null_counts.value(idx)),
                    row_count: row_counts[idx],
                })
                .collect();
            columns.insert(name.clone(), stats);
        }

        Self { columns }
    }

    fn column(&self, name: &str, row_group: usize) -> Option<&ColumnStatistics> {
        self.columns.get(name)?.get(row_group)
    }
}

fn is_float_type(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Float32 | DataType::Float64)
}

fn scalar_at(array: &ArrayRef, index: usize) -> Option<Scalar> {
    if !array.is_valid(index) {
        return None;
    }
    Scalar::try_from_array(array.as_ref(), index).ok()
}

/// A range plus nullability information. `min`/`max` being absent means the
/// non-null value range is unknown, not that the column is all NULL.
#[derive(Clone)]
struct ValueRange {
    min: Option<Scalar>,
    max: Option<Scalar>,
    can_null: bool,
    can_non_null: bool,
}

impl ValueRange {
    fn unknown() -> Self {
        Self {
            min: None,
            max: None,
            can_null: true,
            can_non_null: true,
        }
    }

    fn exact(value: Scalar) -> Self {
        let can_null = value.is_null();
        Self {
            min: Some(value.clone()),
            max: Some(value),
            can_null,
            can_non_null: !can_null,
        }
    }

    fn from_column(stats: &ColumnStatistics) -> Self {
        let all_null = stats.null_count == Some(stats.row_count);
        let no_null = stats.null_count == Some(0);
        Self {
            min: stats.min.clone(),
            max: stats.max.clone(),
            can_null: !no_null,
            can_non_null: !all_null,
        }
    }

    fn singleton(&self) -> Option<&Scalar> {
        let min = self.min.as_ref()?;
        let max = self.max.as_ref()?;
        (min == max).then_some(min)
    }

    fn negate(self) -> Self {
        let Some(value) = self.singleton().cloned() else {
            return Self {
                min: None,
                max: None,
                can_null: self.can_null,
                can_non_null: self.can_non_null,
            };
        };
        let Ok(value) = value.arithmetic_negate() else {
            return Self {
                min: None,
                max: None,
                can_null: self.can_null,
                can_non_null: self.can_non_null,
            };
        };
        Self {
            min: Some(value.clone()),
            max: Some(value),
            can_null: self.can_null,
            can_non_null: self.can_non_null,
        }
    }
}

/// SQL three-valued result possibility. A pruning expression may return NULL;
/// NULL must be treated as "keep" unless TRUE is impossible.
#[derive(Clone, Copy)]
struct BoolPossibility {
    can_true: bool,
    can_false: bool,
    can_null: bool,
}

impl BoolPossibility {
    fn unknown() -> Self {
        Self {
            can_true: true,
            can_false: true,
            can_null: true,
        }
    }

    fn always_true() -> Self {
        Self {
            can_true: true,
            can_false: false,
            can_null: false,
        }
    }

    fn from_value(value: &ValueRange) -> Self {
        let Some(value) = value.singleton() else {
            return Self::unknown();
        };
        match value {
            Scalar::Boolean(Some(value)) => Self {
                can_true: *value,
                can_false: !*value,
                can_null: false,
            },
            Scalar::Boolean(None) => Self {
                can_true: false,
                can_false: false,
                can_null: true,
            },
            _ => Self::unknown(),
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            can_true: self.can_true && rhs.can_true,
            can_false: (self.can_false || rhs.can_false)
                || (self.can_null && rhs.can_false)
                || (rhs.can_null && self.can_false),
            can_null: (self.can_null && !rhs.can_false) || (rhs.can_null && !self.can_false),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            can_true: self.can_true
                || rhs.can_true
                || (self.can_null && rhs.can_true)
                || (rhs.can_null && self.can_true),
            can_false: self.can_false && rhs.can_false,
            can_null: (self.can_null && !rhs.can_true) || (rhs.can_null && !self.can_true),
        }
    }

    fn not(self) -> Self {
        Self {
            can_true: self.can_false,
            can_false: self.can_true,
            can_null: self.can_null,
        }
    }
}

fn possible_predicate(
    expr: &Expr,
    statistics: &FileStatistics,
    row_group: usize,
) -> BoolPossibility {
    match expr {
        Expr::Column(name) => statistics
            .column(name, row_group)
            .map(|stats| BoolPossibility::from_value(&ValueRange::from_column(stats)))
            .unwrap_or_else(BoolPossibility::unknown),
        Expr::Literal(literal) => {
            BoolPossibility::from_value(&ValueRange::exact(literal.value.clone()))
        }
        Expr::BinaryExpr(binary) => match binary.op {
            BinaryOp::And => possible_predicate(&binary.left, statistics, row_group)
                .and(possible_predicate(&binary.right, statistics, row_group)),
            BinaryOp::Or => possible_predicate(&binary.left, statistics, row_group)
                .or(possible_predicate(&binary.right, statistics, row_group)),
            BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
            | BinaryOp::IsDistinctFrom
            | BinaryOp::IsNotDistinctFrom => {
                let left = value_range(&binary.left, statistics, row_group);
                let right = value_range(&binary.right, statistics, row_group);
                compare_ranges(binary.op, left, right)
            }
            _ => BoolPossibility::unknown(),
        },
        Expr::Not(expr) => possible_predicate(expr, statistics, row_group).not(),
        Expr::IsNull(expr) => is_null_possibility(value_range(expr, statistics, row_group), true),
        Expr::IsNotNull(expr) => {
            is_null_possibility(value_range(expr, statistics, row_group), false)
        }
        Expr::InList(in_list) => {
            // DataFusion treats an empty IN list as an unhandled predicate.
            // Keep the row group for both IN () and NOT IN () rather than
            // manufacturing a constant false pruning result.
            if in_list.list.is_empty() {
                return BoolPossibility::always_true();
            }
            let value = value_range(&in_list.expr, statistics, row_group);
            let mut result = BoolPossibility {
                can_true: false,
                can_false: true,
                can_null: false,
            };
            for item in &in_list.list {
                result = result.or(compare_ranges(
                    BinaryOp::Eq,
                    value.clone(),
                    value_range(item, statistics, row_group),
                ));
            }
            if in_list.negated {
                result.not()
            } else {
                result
            }
        }
        Expr::Like(like) => {
            let value = value_range(&like.expr, statistics, row_group);
            let pattern = value_range(&like.pattern, statistics, row_group);
            let Some(pattern) = pattern.singleton() else {
                return BoolPossibility::unknown();
            };
            let Scalar::Utf8(Some(pattern)) = pattern else {
                return BoolPossibility::unknown();
            };
            if like.case_insensitive {
                return BoolPossibility::unknown();
            }

            let (prefix, rest) = split_like_prefix(pattern);
            if prefix.is_empty() && !rest.is_empty() {
                return BoolPossibility::unknown();
            }
            if rest.is_empty() {
                let result = compare_ranges(
                    BinaryOp::Eq,
                    value,
                    ValueRange::exact(Scalar::Utf8(Some(prefix))),
                );
                return if like.negated { result.not() } else { result };
            }
            let Some(upper) = increment_utf8(&prefix) else {
                return BoolPossibility::unknown();
            };
            let lower = ValueRange::exact(Scalar::Utf8(Some(prefix.clone())));
            let upper = ValueRange::exact(Scalar::Utf8(Some(upper)));
            let matches_prefix = compare_ranges(BinaryOp::GtEq, value.clone(), lower)
                .and(compare_ranges(BinaryOp::LtEq, value.clone(), upper));
            if !like.negated {
                matches_prefix
            } else if rest == "%" {
                if !value.can_non_null {
                    BoolPossibility {
                        can_true: false,
                        can_false: false,
                        can_null: value.can_null,
                    }
                } else if bounds_match_prefix(&value, &prefix) {
                    BoolPossibility {
                        can_true: false,
                        can_false: value.can_non_null,
                        can_null: value.can_null,
                    }
                } else {
                    BoolPossibility::unknown()
                }
            } else {
                BoolPossibility::unknown()
            }
        }
        Expr::Cast(cast) => possible_predicate(&cast.expr, statistics, row_group),
        Expr::TryCast(cast) => possible_predicate(&cast.expr, statistics, row_group),
        Expr::Negative(_) | Expr::Case(_) | Expr::Function(_) => BoolPossibility::unknown(),
    }
}

fn value_range(expr: &Expr, statistics: &FileStatistics, row_group: usize) -> ValueRange {
    match expr {
        Expr::Column(name) => statistics
            .column(name, row_group)
            .map(ValueRange::from_column)
            .unwrap_or_else(ValueRange::unknown),
        Expr::Literal(Literal { value }) => ValueRange::exact(value.clone()),
        Expr::Negative(expr) => value_range(expr, statistics, row_group).negate(),
        Expr::Cast(cast) => cast_range(
            value_range(&cast.expr, statistics, row_group),
            &cast.cast_type,
            false,
        ),
        Expr::TryCast(cast) => cast_range(
            value_range(&cast.expr, statistics, row_group),
            &cast.cast_type,
            true,
        ),
        Expr::BinaryExpr(binary) => arithmetic_range(binary, statistics, row_group),
        _ => ValueRange::unknown(),
    }
}

fn cast_range(value: ValueRange, target: &DataType, is_try_cast: bool) -> ValueRange {
    let (Some(min), Some(max)) = (value.min.clone(), value.max.clone()) else {
        return ValueRange {
            // A TRY CAST can turn a non-NULL input into NULL when conversion
            // fails. Never claim that the result must be non-NULL.
            can_null: value.can_null || (is_try_cast && value.can_non_null),
            can_non_null: value.can_non_null,
            ..ValueRange::unknown()
        };
    };

    // Endpoint casting is only exact when the cast does not change the
    // physical type. Narrowing, string parsing, and timestamp-unit casts can
    // otherwise change ordering or overflow; treating those bounds as unknown
    // is conservative and keeps the row-level predicate authoritative.
    if min.data_type() != *target || max.data_type() != *target {
        return ValueRange {
            // Both CAST and TRY CAST can change the meaning of NULLability:
            // CAST may fail at evaluation time, while TRY CAST produces NULL.
            // Keeping NULL possible avoids pruning rows that cannot be proved
            // impossible from the source min/max alone.
            can_null: value.can_null || value.can_non_null,
            can_non_null: value.can_non_null,
            ..ValueRange::unknown()
        };
    }
    ValueRange {
        min: Some(min),
        max: Some(max),
        can_null: value.can_null,
        can_non_null: value.can_non_null,
    }
}

fn arithmetic_range(
    binary: &BinaryExpr,
    statistics: &FileStatistics,
    row_group: usize,
) -> ValueRange {
    let left = value_range(&binary.left, statistics, row_group);
    let right = value_range(&binary.right, statistics, row_group);
    let can_null = left.can_null || right.can_null;
    let can_non_null = left.can_non_null && right.can_non_null;

    // The expression evaluator uses wrapping integer arithmetic. Interval
    // endpoint arithmetic is therefore unsafe for non-singleton ranges (and
    // for division by a range that may contain zero). Only fold exact scalar
    // operands; all other arithmetic remains unknown.
    let (Some(left), Some(right)) = (left.singleton().cloned(), right.singleton().cloned()) else {
        return ValueRange {
            min: None,
            max: None,
            can_null,
            can_non_null,
        };
    };
    let expr = Expr::BinaryExpr(BinaryExpr {
        left: Box::new(Expr::Literal(Literal { value: left })),
        op: binary.op,
        right: Box::new(Expr::Literal(Literal { value: right })),
    });
    let Ok(value) = expr.constant_eval() else {
        return ValueRange {
            min: None,
            max: None,
            can_null,
            can_non_null,
        };
    };
    let result = ValueRange::exact(value);
    ValueRange {
        min: result.min,
        max: result.max,
        can_null: can_null || result.can_null,
        can_non_null: can_non_null && result.can_non_null,
    }
}

fn is_null_possibility(value: ValueRange, is_null: bool) -> BoolPossibility {
    let can_true = if is_null {
        value.can_null
    } else {
        value.can_non_null
    };
    let can_false = if is_null {
        value.can_non_null
    } else {
        value.can_null
    };
    BoolPossibility {
        can_true,
        can_false,
        can_null: false,
    }
}

fn compare_ranges(op: BinaryOp, left: ValueRange, right: ValueRange) -> BoolPossibility {
    // First evaluate the comparison over the possible non-null values. If one
    // side is known to be all NULL, there is no non-null comparison result.
    let non_null = if left.can_non_null && right.can_non_null {
        let equality = equality_possibility(&left, &right);
        match op {
            BinaryOp::Eq => equality,
            BinaryOp::NotEq => equality.not(),
            BinaryOp::IsDistinctFrom => equality.not(),
            BinaryOp::IsNotDistinctFrom => equality,
            _ => ordered_relation(op, &left, &right),
        }
    } else {
        BoolPossibility {
            can_true: false,
            can_false: false,
            can_null: false,
        }
    };

    // IS DISTINCT FROM is not nullable: NULL/NULL is false and NULL/non-NULL
    // is true. The ordinary comparison operators instead produce NULL when
    // either operand is NULL, which is treated as keep by the caller.
    if op == BinaryOp::IsDistinctFrom || op == BinaryOp::IsNotDistinctFrom {
        let both_null = left.can_null && right.can_null;
        let one_null =
            (left.can_null && right.can_non_null) || (left.can_non_null && right.can_null);
        if op == BinaryOp::IsDistinctFrom {
            return BoolPossibility {
                can_true: non_null.can_false || one_null,
                can_false: non_null.can_true || both_null,
                can_null: false,
            };
        }
        return BoolPossibility {
            can_true: non_null.can_true || both_null,
            can_false: non_null.can_false || one_null,
            can_null: false,
        };
    }

    BoolPossibility {
        can_true: non_null.can_true,
        can_false: non_null.can_false,
        can_null: (left.can_null && (right.can_null || right.can_non_null))
            || (right.can_null && (left.can_null || left.can_non_null)),
    }
}

fn equality_possibility(left: &ValueRange, right: &ValueRange) -> BoolPossibility {
    let can_true = match (&left.min, &left.max, &right.min, &right.max) {
        (Some(lmin), Some(lmax), Some(rmin), Some(rmax)) => {
            let Some(left_before_right) = try_is_less(lmax, rmin) else {
                return BoolPossibility::unknown();
            };
            let Some(right_before_left) = try_is_less(rmax, lmin) else {
                return BoolPossibility::unknown();
            };
            !left_before_right && !right_before_left
        }
        _ => true,
    };
    let can_false = match (left.singleton(), right.singleton()) {
        (Some(left), Some(right)) => try_is_equal(left, right)
            .map(|equal| !equal)
            .unwrap_or(true),
        _ => true,
    };
    BoolPossibility {
        can_true,
        can_false,
        can_null: false,
    }
}

fn ordered_relation(op: BinaryOp, left: &ValueRange, right: &ValueRange) -> BoolPossibility {
    let (Some(lmin), Some(lmax), Some(rmin), Some(rmax)) =
        (&left.min, &left.max, &right.min, &right.max)
    else {
        return BoolPossibility {
            can_true: true,
            can_false: true,
            can_null: false,
        };
    };

    let lmin_rmax = try_is_less(lmin, rmax);
    let rmax_lmin = try_is_less(rmax, lmin);
    let lmax_rmin = try_is_less(lmax, rmin);
    let rmin_lmax = try_is_less(rmin, lmax);
    if [lmin_rmax, rmax_lmin, lmax_rmin, rmin_lmax]
        .iter()
        .any(Option::is_none)
    {
        return BoolPossibility::unknown();
    }
    let lmin_rmax = lmin_rmax.unwrap();
    let rmax_lmin = rmax_lmin.unwrap();
    let lmax_rmin = lmax_rmin.unwrap();
    let rmin_lmax = rmin_lmax.unwrap();

    let can_true = match op {
        BinaryOp::Lt => lmin_rmax,
        BinaryOp::LtEq => !rmax_lmin,
        BinaryOp::Gt => rmin_lmax,
        BinaryOp::GtEq => !lmax_rmin,
        _ => true,
    };
    let can_false = match op {
        BinaryOp::Lt => !lmax_rmin,
        BinaryOp::LtEq => rmin_lmax,
        BinaryOp::Gt => !rmax_lmin,
        BinaryOp::GtEq => lmin_rmax,
        _ => true,
    };
    BoolPossibility {
        can_true,
        can_false,
        can_null: false,
    }
}

fn bounds_match_prefix(value: &ValueRange, prefix: &str) -> bool {
    let (Some(Scalar::Utf8(Some(min))), Some(Scalar::Utf8(Some(max)))) =
        (value.min.as_ref(), value.max.as_ref())
    else {
        return false;
    };
    min.starts_with(prefix) && max.starts_with(prefix)
}

fn split_like_prefix(pattern: &str) -> (String, &str) {
    let mut prefix = String::with_capacity(pattern.len());
    let mut chars = pattern.char_indices();
    while let Some((idx, ch)) = chars.next() {
        match ch {
            '%' | '_' => return (prefix, &pattern[idx..]),
            '\\' => match chars.next() {
                Some((_, escaped)) => prefix.push(escaped),
                None => prefix.push('\\'),
            },
            _ => prefix.push(ch),
        }
    }
    (prefix, "")
}

fn increment_utf8(value: &str) -> Option<String> {
    let mut chars = value.chars().collect::<Vec<_>>();
    for index in (0..chars.len()).rev() {
        let codepoint = chars[index] as u32;
        let next_codepoint = codepoint + 1;
        if next_codepoint == 0xfffe || next_codepoint == 0xffff {
            continue;
        }
        let Some(next) = char::from_u32(next_codepoint) else {
            continue;
        };
        chars[index] = next;
        chars.truncate(index + 1);
        return Some(chars.into_iter().collect());
    }
    None
}

fn try_is_less(left: &Scalar, right: &Scalar) -> Option<bool> {
    try_ordering(left, right).map(|ordering| ordering == std::cmp::Ordering::Less)
}

fn try_is_equal(left: &Scalar, right: &Scalar) -> Option<bool> {
    try_ordering(left, right).map(|ordering| ordering == std::cmp::Ordering::Equal)
}

fn try_ordering(left: &Scalar, right: &Scalar) -> Option<std::cmp::Ordering> {
    // Scalar::partial_cmp intentionally compares raw values within a variant
    // for some types (notably timestamps with different timezones). IndexLake
    // has no automatic type coercion, so require exact Arrow type equality
    // before using an ordering for pruning.
    if left.data_type() != right.data_type() {
        return None;
    }
    // Unlike `f64::total_cmp`, Arrow comparisons treat NaN as unordered and
    // NaN equality as false. Return unknown rather than inventing a total
    // order for pruning.
    if matches!(left, Scalar::Float32(Some(v)) if v.is_nan())
        || matches!(right, Scalar::Float32(Some(v)) if v.is_nan())
        || matches!(left, Scalar::Float64(Some(v)) if v.is_nan())
        || matches!(right, Scalar::Float64(Some(v)) if v.is_nan())
    {
        return None;
    }
    left.partial_cmp(right)
}

/// Number of suffix bytes to prefetch for the footer: size/12 (DataFusion's
/// default heuristic), at least 64 KiB, clamped to the file size.
pub(crate) fn footer_size_hint(size: u64) -> usize {
    std::cmp::min(size as usize, std::cmp::max(size as usize / 12, 64 * 1024))
}

/// Read and decode the footer metadata through an already opened input file.
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
    if n < 8 || &suffix[n - 4..] != b"PAR1" {
        return Err(ILError::internal(format!(
            "Parquet file {relative_path} has an invalid footer trailer"
        )));
    }
    let footer_len = u32::from_le_bytes(suffix[n - 8..n - 4].try_into().unwrap()) as u64;
    if footer_len + 8 > size {
        return Err(ILError::internal(format!(
            "Parquet file {relative_path} has an invalid footer length"
        )));
    }
    let footer_len = footer_len as usize;
    let metadata = if footer_len <= n - 8 {
        suffix.slice(n - 8 - footer_len..n - 8)
    } else {
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
    use std::sync::Arc;

    use arrow::array::{Float64Array, Int64Array, RecordBatch, StringArray, TimestampSecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use parquet::arrow::arrow_reader::RowSelection;
    use parquet::arrow::arrow_writer::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;
    use crate::expr::{InList, TryCast, col, lit};

    fn write_test_parquet(schema: &SchemaRef, columns: Vec<ArrayRef>) -> ParquetMetaData {
        let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(10))
            .build();
        let mut writer =
            ArrowWriter::try_new(Vec::<u8>::new(), schema.clone(), Some(props)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap()
    }

    fn int_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("grp", DataType::Int64, false)]))
    }

    fn int_pruner(expr: Expr) -> Option<RowGroupPruner> {
        let schema = int_schema();
        build_row_group_pruner(&[expr], &schema)
    }

    #[test]
    fn prunes_matching_row_groups() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let pruner = int_pruner(col("grp").eq(lit(25i64)));
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 10),
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn skips_file_when_no_group_can_match() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let pruner = int_pruner(col("grp").eq(lit(500i64)));
        assert!(matches!(
            prune_file_row_groups(&metadata, pruner.as_ref(), 30),
            Some(PruneOutcome::Skip)
        ));
    }

    #[test]
    fn respects_sql_null_semantics() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            true,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(2)])) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("value").is_null()], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 3).is_none());
    }

    #[test]
    fn supports_float_and_string_ranges() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Float64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let metadata = write_test_parquet(
            &schema,
            vec![
                Arc::new(Float64Array::from_iter_values((0..30).map(|v| v as f64))) as ArrayRef,
                Arc::new(StringArray::from_iter_values(
                    (0..30).map(|i| format!("key_{i:03}")),
                )) as ArrayRef,
            ],
        );
        let filters = vec![
            col("value").eq(lit(25.5f64)),
            col("name").eq(lit("key_025")),
        ];
        let pruner = build_row_group_pruner(&filters, &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 10),
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn float_statistics_are_not_used_for_pruning() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Float64,
            false,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Float64Array::from_iter_values([
                1.0,
                f64::NAN,
                -0.0,
                0.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ])) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("value").gt(lit(10.0f64))], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 10).is_none());
    }

    #[test]
    fn try_cast_failures_are_treated_as_null() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Utf8,
            false,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                "not-a-number",
                10,
            ))) as ArrayRef],
        );
        let filter = Expr::IsNull(Box::new(Expr::TryCast(TryCast {
            expr: Box::new(col("value")),
            cast_type: DataType::Int64,
        })));
        let pruner = build_row_group_pruner(&[filter], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 10).is_none());
    }

    #[test]
    fn timestamp_timezone_mismatch_disables_pruning() {
        let timezone: Arc<str> = Arc::from("+08:00");
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Timestamp(TimeUnit::Second, Some(timezone.clone())),
            false,
        )]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(
                TimestampSecondArray::from_iter_values(0..30).with_timezone_opt(Some(timezone)),
            ) as ArrayRef],
        );
        let filter = Expr::BinaryExpr(BinaryExpr {
            left: Box::new(col("value")),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal(Literal {
                value: Scalar::TimestampSecond(Some(25), Some(Arc::from("UTC"))),
            })),
        });
        let pruner = build_row_group_pruner(&[filter], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
    }

    #[test]
    fn nan_scalar_ordering_is_unknown() {
        let nan = Scalar::Float64(Some(f64::NAN));
        assert_eq!(try_is_equal(&nan, &nan), None);
        assert_eq!(try_is_less(&nan, &Scalar::Float64(Some(1.0))), None);
        assert_eq!(try_is_less(&Scalar::Float64(Some(1.0)), &nan), None);
    }

    #[test]
    fn incomparable_filter_types_disable_pruning() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let pruner = int_pruner(col("grp").gt(lit(5i32)));
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
    }

    #[test]
    fn empty_in_lists_are_conservative() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        for negated in [false, true] {
            let filter = Expr::InList(InList {
                expr: Box::new(col("grp")),
                list: vec![],
                negated,
            });
            let pruner = build_row_group_pruner(&[filter], &schema);
            assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
        }
    }

    #[test]
    fn prunes_with_in_list() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let filter = Expr::InList(InList {
            expr: Box::new(col("grp")),
            list: vec![lit(25i64), lit(26i64)],
            negated: false,
        });
        let pruner = build_row_group_pruner(&[filter], &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 10),
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn prunes_not_in_when_a_group_is_all_equal() {
        let schema = int_schema();
        let values = [1i64; 10].into_iter().chain([2i64; 10]).chain([3i64; 10]);
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(values)) as ArrayRef],
        );
        let filter = Expr::InList(InList {
            expr: Box::new(col("grp")),
            list: vec![lit(2i64)],
            negated: true,
        });
        let pruner = build_row_group_pruner(&[filter], &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 20),
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn like_without_wildcards_uses_string_bounds() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key-{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").like(lit("key-025"))], &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 10),
            _ => panic!("expected partial pruning"),
        }
    }

    #[test]
    fn not_like_prefix_keeps_mixed_endpoint_group() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let values = std::iter::repeat_n("foo-a", 5).chain(std::iter::repeat_n("fop-z", 5));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(values)) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").not_like(lit("foo%"))], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 10).is_none());
    }

    #[test]
    fn not_like_without_wildcards_uses_not_equal_bounds() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key-{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").not_like(lit("key-025"))], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
    }

    #[test]
    fn like_prefix_uses_string_range() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key-{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").like(lit("key-02%"))], &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 10),
            _ => panic!("expected prefix pruning"),
        }
    }

    #[test]
    fn like_with_leading_wildcard_is_conservative() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key-{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").like(lit("%02"))], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
    }

    #[test]
    fn not_like_prefix_prunes_all_matching_group() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key-{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").not_like(lit("key-02%"))], &schema);
        match prune_file_row_groups(&metadata, pruner.as_ref(), 30) {
            Some(PruneOutcome::Partial(selection)) => assert_eq!(selection.row_count(), 20),
            _ => panic!("expected NOT LIKE prefix pruning"),
        }
    }

    #[test]
    fn ilike_is_conservative() {
        let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(StringArray::from_iter_values(
                (0..30).map(|i| format!("key_{i:03}")),
            )) as ArrayRef],
        );
        let pruner = build_row_group_pruner(&[col("name").ilike(lit("KEY_025"))], &schema);
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 30).is_none());
    }

    #[test]
    fn mismatch_disables_pruning() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let pruner = int_pruner(col("grp").eq(lit(25i64)));
        assert!(prune_file_row_groups(&metadata, pruner.as_ref(), 29).is_none());
    }

    #[test]
    fn deleted_selection_stays_empty() {
        let schema = int_schema();
        let metadata = write_test_parquet(
            &schema,
            vec![Arc::new(Int64Array::from_iter_values(0..30)) as ArrayRef],
        );
        let pruner = int_pruner(col("grp").eq(lit(25i64)));
        let Some(PruneOutcome::Partial(selection)) =
            prune_file_row_groups(&metadata, pruner.as_ref(), 30)
        else {
            panic!("expected partial pruning");
        };
        let deleted = RowSelection::from_consecutive_ranges(std::iter::empty(), 30);
        assert_eq!(deleted.intersection(&selection).row_count(), 0);
    }
}
