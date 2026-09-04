# Row-group pruning 设计与实现

本文说明 `indexlake` 当前基于 Parquet footer statistics 的 row-group pruning 设计。目标只有一个：**只有能证明一个 row group 中不可能存在满足 filter 的行时，才允许跳过它**。因此所有缺失、错误、类型不匹配或语义不确定的情况都必须保守地保留 row group。

## 1. 总体原则

1. **无 false negative 优先于剪枝率**
   - Pruning 只能跳过确定不可能返回 `TRUE` 的 row group。
   - Predicate 对某一行可能返回 `TRUE`、`NULL`，或实现无法判断时，row group 都必须保留。
   - Row-level filter 仍是最终过滤器；statistics pruning 只是减少读取和解码量的安全前置判断。

2. **SQL 三值布尔语义**
   - Predicate 不是普通 boolean，而是包含 `TRUE` / `FALSE` / `NULL` 可能性的三值结果。
   - `AND`、`OR`、`NOT` 都按 SQL Kleene logic 处理。
   - 只有 `can_true == false` 时才能剪掉 row group。

3. **不依赖 DataFusion**
   - `indexlake` core 自己维护表达式树和 scalar 类型。
   - 这里参考 DataFusion pruning 的语义，但不引入 DataFusion normal dependency，也不复制其 coercion/physical expression 行为。
   - core API 没有自动 type coercion，因此类型不完全一致时直接返回 unknown。

## 2. 组件结构

### `RowGroupPruner`

- 在 scan 开始时由 `build_row_group_pruner(&filters, &table_schema.arrow_schema)` 构建一次。
- 只包含 scan filters 和 table schema 的不可变数据，内部使用 `Arc`，可跨 file 和 async task 安全 clone。
- 每个 file 的 footer 到达后，再对该 file 的每个 row group 评估同一份 predicate。

### `FileStatistics`

- 使用 `parquet::arrow::arrow_reader::statistics::StatisticsConverter` 将 Parquet footer min/max/null count 转成 Arrow arrays。
- 每个 required column 只转换一次，避免每个 row group 重复解码 metadata。
- Column stats 保存：
  - `min`
  - `max`
  - `null_count`
  - `row_count`
- 某一列转换失败、字段缺失或统计异常时，不报错、不剪枝，而是让该列值域保持 unknown。

### `ValueRange`

`ValueRange` 表示一个非空值区间加上 null 可能性：

```text
min: Option<Scalar>
max: Option<Scalar>
can_null: bool
can_non_null: bool
```

- `min/max` 缺失表示非 NULL 值域未知，不表示全 NULL。
- 只有 `null_count == 0` 才能确定 `can_null == false`。
- 只有 `null_count == row_count` 才能确定 `can_non_null == false`。
- Literal 形成单点区间。
- Column statistics 形成闭区间 `[min, max]`。

### `BoolPossibility`

```text
can_true
can_false
can_null
```

顶层 fold 规则：

```text
filter[0].possibility
    .and(filter[1].possibility)
    .and(filter[2].possibility)
    ...
```

只要最终 `can_true == false`，即可证明没有任何行能让 WHERE 条件为 `TRUE`，从而剪掉 row group。

## 3. 安全入口与坐标约束

`prune_file_row_groups` 的入口有硬性检查：

```text
footer_row_group_total == catalog record_count
```

如果不一致，立即放弃 pruning。

原因：catalog 中的 delete bitmap / row validity 使用 catalog record count 对应的文件行坐标；row group selection 使用 footer row group 行坐标。两者不一致时，selection 可能错位，甚至复活已删除行。

Pruning 返回三种结果：

- `None`：不剪枝，继续使用原始 delete validity selection。
- `PruneOutcome::Skip`：所有 row group 都不可能命中，整个 file 跳过。
- `PruneOutcome::Partial(selection)`：只读取保留下来的 row group。

在 Parquet read path 中：

1. 每个 file 只读取一次 footer metadata。
2. 该 metadata 同时用于 pruning 和 `ParquetRecordBatchStreamBuilder`。
3. 先计算 row-group selection。
4. 与 catalog delete validity selection 做 intersection。
5. 交给 Parquet reader。
6. 后续仍应用原表达式 row filter。

这意味着 delete bitmap 和 row-group pruning 的坐标始终在同一份 footer 行数网格上。

## 4. 表达式语义

### 比较操作

- `EQ`：两个区间可能相交，则 `can_true`；两侧 singleton 且明确不同，则 `can_false`。
- `NOT_EQ`：对 `EQ` 取反。
- `LT/LT_EQ/GT/GT_EQ`：使用区间端点判断可能关系。
- 跨类型、跨 timezone、跨 decimal precision/scale 等无法安全比较时返回 unknown。
- `Float32/Float64` 的 `NaN` 一律 unordered，普通排序判断返回 unknown。

普通 SQL comparison 中任一侧为 `NULL` 时结果是 `NULL`，因此 `can_null` 必须保留。

### `IS DISTINCT FROM` / `IS NOT DISTINCT FROM`

这两个操作不是 nullable：

- `NULL IS DISTINCT FROM NULL` 是 false。
- `NULL IS DISTINCT FROM non-null` 是 true。
- `NULL IS NOT DISTINCT FROM NULL` 是 true。
- `NULL IS NOT DISTINCT FROM non-null` 是 false。

因此先评估非 NULL 区间关系，再叠加 NULL/NULL 或 NULL/non-NULL 的情况。

### `IS NULL` / `IS NOT NULL`

- 只有 null count 明确时才收紧 nullability。
- `null_count == row_count` 时确定全 NULL。
- `null_count == 0` 时确定全非 NULL。
- null count 缺失时，`IS NULL` 和 `IS NOT NULL` 都保持可能，row group 保留。

### `IN` / `NOT IN`

`IN` 展开为：

```text
value = item[0] OR value = item[1] OR ...
```

`NOT IN` 再取反。

特殊规则：

- 空 `IN ()` 保守保留。
- 空 `NOT IN ()` 保守保留。
- list 中存在 NULL literal 时，SQL Kleene 语义可能产生 `NULL`，不能据此剪枝。
- 任意 item 与值域不可比较时，整个结果进入 unknown。

### `LIKE`

当前支持安全的固定 prefix 优化：

```sql
col LIKE 'prefix%'
```

转换成：

```text
col >= 'prefix' AND col <= successor('prefix')
```

规则：

- Pattern 是 literal 才参与 pruning。
- Case-insensitive `ILIKE` 保守保留。
- leading `%` 或 `_` 保守保留。
- `prefix` 内的 SQL wildcard escape 会被解码。
- `successor()` 按 Unicode scalar / UTF-8 ordering 递增；无法构造安全 successor 时保守保留。
- Upper bound 使用 inclusive comparison；即使 successor 本身可能出现，也只会多保留，不会误剪。
- `NOT LIKE 'prefix%'` 只有在 row group 的 min 和 max 都明确以同一 prefix 开头时才证明所有非 NULL 值都匹配 prefix，因此 predicate 不可能为 TRUE。
- 其它中间 wildcard、`NOT LIKE` 复杂 pattern、大小写不敏感 pattern 都保守保留。

### `CAST` / `TRY_CAST`

Pruning 对 cast 很保守。

- 只有 endpoint scalar 的物理 Arrow type 与 target type 完全相同时，才保留原 min/max。
- 改变类型时不推断新值域，返回 unknown。
- 改变类型时强制保留 NULL 可能性：
  - `CAST` 可能在 row-level evaluation 报错，不能据此证明不可能。
  - `TRY_CAST` 失败会返回 NULL，因此即使源列全非 NULL，结果也可能为 NULL。

例如：

```sql
IS NULL(TRY_CAST(string_col AS INT))
```

即使 `string_col` 的 group 内全是非 NULL 字符串，只要转换可能失败，row group 必须保留。

### `Negative` 和 arithmetic

- `Negative` 只对 singleton 做 exact negation。
- Arithmetic 只对两侧都 singleton 的 exact scalar fold。
- Range arithmetic、除零、溢出、除数可能为零等情况一律 unknown。
- Row-level evaluator 使用 wrapping integer arithmetic，pruning 不假设 endpoint arithmetic 一定保留区间关系。

### 其它表达式

`CASE`、function、unsupported binary operator 等都返回 unknown。

## 5. Float statistics 特例

Parquet writer 可能从 float min/max statistics 中排除 `NaN`，而 Arrow / SQL comparison 的 `NaN` 行为与普通 total order 不同。直接使用这些 min/max 会误判包含 `NaN` 的 row group。

因此当前实现：

- `Float32` / `Float64` column 的 min/max 不用于 pruning。
- 仍尽量使用 null count 提供的 nullability 信息。
- Float literal 或 expression 中出现 `NaN` 时，ordering/equality 判断返回 unknown。

这样可能减少剪枝机会，但避免 false negative。

## 6. Row-group selection 与 read path

单个 Parquet file 的读取流程如下：

```text
open file
  -> read footer once
  -> build ArrowReaderMetadata
  -> compute projection
  -> evaluate row-group pruning
  -> intersect pruning selection with delete validity selection
  -> build ParquetRecordBatchStream
  -> apply original ExprPredicate row filter
```

`TablePartitionScanner` 在 scan 创建时构建一次 pruner。多个 data file 共享同一个 pruner，避免每个 file 重复构建表达式结构。

## 7. 测试覆盖

核心单测位于 `indexlake/src/storage/prune.rs`，覆盖：

- 基础 row group 剪枝。
- 整 file skip。
- SQL NULL 语义。
- Float statistics 不参与比较。
- `NaN` ordering unknown。
- 跨类型 / timezone mismatch 保守保留。
- `TRY_CAST` 失败值保留。
- `IN` / `NOT IN`。
- 空 `IN ()` / `NOT IN ()`。
- LIKE prefix。
- leading wildcard。
- ILIKE 保守保留。
- `NOT LIKE` prefix 和 mixed endpoint。
- footer row count 与 catalog record count mismatch。
- pruning selection 与 delete selection intersection。

## 8. 已知边界

以下能力当前刻意不启用或保守处理：

- Float min/max statistics pruning。
- Page index / offset index pruning。
- Bloom filter pruning。
- 跨 footer/catalog 行数不一致时的剪枝。
- 需要 type coercion 的比较。
- 复杂 `NOT LIKE` pattern。
- 大小写不敏感 LIKE。
- Range arithmetic。
- `CASE` / function 内部推导。

这些点未来可以增量增强，但必须保持相同约束：**无法证明 predicate 不可能为 TRUE 时必须保留 row group。**
