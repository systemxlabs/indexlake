# Index 动态列技术方案（Table Scan + Table Search）

## 1. 背景

`indexlake` 当前已经支持 index 参与 `table.search()` 排序，以及参与 `table.scan()` 过滤，但 index 产生的附加信息还不能作为结果集列返回。

典型例子：

- BM25 在搜索时已经能计算每条命中的 `score`
- `table/search.rs` 已经会使用 `score` 做全局排序
- 结果集最终只返回表字段，不返回 `score`

同时，`table.scan()` 也已经存在索引过滤路径：

- `table/scan.rs` 会把部分 filter 分配给 index
- index 返回命中的 `row_id`
- 表层再回表取出真正的数据行

但 scan 路径同样没有“索引给结果集动态加列”的能力。

本方案目标是统一 `TableScan` 和 `TableSearch` 两条链路，让 index 都可以按需给结果集添加动态列。BM25 的 `score` 只是第一批能力，不应作为特例硬编码在表层。

## 2. 现状

### 2.1 `table.search()` 现状

[indexlake/src/table/search.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/search.rs) 当前流程：

1. 根据 `SearchQuery.index_kind()` 选出目标 index
2. 分别搜索 inline rows 和 data files
3. 合并 `SearchIndexEntries`
4. 回表读取基础列
5. 按 `score` 排序
6. 返回 `RecordBatch`

当前索引搜索结果定义位于 [indexlake/src/index/mod.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/index/mod.rs)：

```rust
pub struct SearchIndexEntries {
    pub row_id_scores: Vec<RowIdScore>,
    pub score_higher_is_better: bool,
}
```

这只能表达：

- 命中了哪些行
- 这些行的排序分值

不能表达：

- 需要返回哪些索引附加列
- 附加列的数据类型
- 附加列的别名

### 2.2 `table.scan()` 现状

[indexlake/src/table/scan.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/scan.rs) 当前流程分两类：

- 无索引过滤：走普通表扫描
- 有索引过滤：先由 index 过滤出 `row_id`，再回表读取匹配行

当前索引过滤结果定义同样位于 [indexlake/src/index/mod.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/index/mod.rs)：

```rust
pub struct FilterIndexEntries {
    pub row_ids: Vec<Uuid>,
}
```

这同样无法表达动态列。

另外，当前 `process_index_scan()` 直接拼接 inline/data-file stream，没有复用 `TablePartitionScanner` 的全局 `offset/limit` 逻辑。这意味着如果要在 index scan 路径增加动态列，顺带需要把结果整流逻辑一起补齐，否则行为不一致。

### 2.3 Schema 侧已有基础

[indexlake/src/utils.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/utils.rs) 中的 `correct_batch_schema()` 会把内部 field id 转回业务列名，但对普通字段名原样透传。

因此，只要结果 batch 里追加的动态列名不是内部 field id，就不会在 schema 更正阶段被破坏。

## 3. 目标与非目标

### 3.1 目标

1. `TableScan` 和 `TableSearch` 都支持 index 动态添加结果列
2. 动态列必须显式请求，不默认返回
3. 动态列用 Arrow 列返回，不走逐行对象拼装
4. 表层不为 BM25/HNSW/BTree/RStar 写硬编码分支
5. 不请求动态列时，现有行为保持不变

### 3.2 非目标

1. 本期不要求 DataFusion SQL 立刻暴露动态列
2. 本期不支持动态列参与 `update/delete`
3. 本期不设计“任意 index 在没有参与 scan/search 的情况下仍可独立产出全表列”
4. 本期不做多 index 联合 search

## 4. 设计原则

1. 按需返回：调用方显式请求，index 才计算和返回
2. index 自治：列名语义、类型、生成逻辑由 index 自己定义
3. 表层通用：`table/scan.rs` 和 `table/search.rs` 只做调度、校验、拼装
4. 列式优先：动态列保持 Arrow 原生列式表示
5. 向后兼容：默认请求为空时，输出 schema 与现在一致

## 5. 对外 API 设计

### 5.1 统一的动态列请求结构

新增统一请求结构：

```rust
pub struct IndexColumnRequest {
    pub index_name: Option<String>,
    pub name: String,
    pub alias: Option<String>,
}
```

字段含义：

- `index_name`：动态列来自哪个 index
- `name`：index 内部定义的逻辑列名，例如 `score`、`distance`
- `alias`：结果集中的输出列名；为空时默认使用 `name`

### 5.2 挂到 `TableSearch`

```rust
pub struct TableSearch {
    pub query: Arc<dyn SearchQuery>,
    pub projection: Option<Vec<usize>>,
    pub index_columns: Vec<IndexColumnRequest>,
}
```

校验规则：

- `search` 本身只会命中一个目标 index
- 如果 `index_name` 为空，默认绑定到这个目标 index
- 如果 `index_name` 不为空，必须等于目标 index 名称，否则报错

### 5.3 挂到 `TableScan`

```rust
pub struct TableScan {
    pub projection: Option<Vec<usize>>,
    pub filters: Vec<Expr>,
    pub batch_size: usize,
    pub partition: TableScanPartition,
    pub offset: usize,
    pub limit: Option<usize>,
    pub index_columns: Vec<IndexColumnRequest>,
}
```

校验规则：

- `scan` 允许多个 index 同时参与过滤
- `index_columns` 中的每一项必须能解析到某个实际参与本次 scan 的 index
- 当 `index_name` 为空且本次 scan 参与过滤的 index 不唯一时，报错
- 当请求的列来自未参与本次 scan 的 index 时，报错

这样做的原因很直接：scan 场景下 index 来源可能不唯一，必须在入口消除歧义。

## 6. Index 层扩展

### 6.1 结果列结构

新增 index 输出列结构：

```rust
pub struct IndexResultColumn {
    pub field: FieldRef,
    pub values: ArrayRef,
}
```

约束：

- `values.len()` 必须与当前结果集中行数一致
- `field.name()` 为最终输出列名，已经处理 alias
- `field.data_type()` 由 index 自己定义

### 6.2 统一的结果列请求

表层会把全局的 `IndexColumnRequest` 解析成“发给单个 index 的请求”：

```rust
pub struct RequestedIndexColumn {
    pub name: String,
    pub output_name: String,
}

pub struct IndexResultOptions {
    pub columns: Vec<RequestedIndexColumn>,
}
```

这样 `Index` trait 不需要理解跨 index 的全局请求，只需要处理“当前这个 index 要返回哪些列”。

### 6.3 扩展 `Index` trait

```rust
#[async_trait::async_trait]
pub trait Index {
    async fn search(
        &self,
        query: &dyn SearchQuery,
        options: &IndexResultOptions,
    ) -> ILResult<SearchIndexEntries>;

    async fn filter(
        &self,
        filters: &[Expr],
        options: &IndexResultOptions,
    ) -> ILResult<FilterIndexEntries>;
}
```

### 6.4 扩展返回结构

```rust
pub struct SearchIndexEntries {
    pub row_id_scores: Vec<RowIdScore>,
    pub score_higher_is_better: bool,
    pub dynamic_columns: Vec<IndexResultColumn>,
}

pub struct FilterIndexEntries {
    pub row_ids: Vec<Uuid>,
    pub dynamic_columns: Vec<IndexResultColumn>,
}
```

说明：

- `search` 场景仍保留 `row_id_scores` 作为排序依据
- `filter` 场景没有统一排序语义，因此只返回 `row_ids`
- 两者都允许返回动态列

## 7. `TableSearch` 执行方案

### 7.1 入口解析

在 [indexlake/src/table/search.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/search.rs) 中：

1. 先解析目标 search index
2. 校验 `index_columns`
3. 生成该 index 的 `IndexResultOptions`

### 7.2 索引搜索阶段

`search_inline_rows()` 和 `search_index_file()` 调用 `Index::search(query, options)`。

index 返回：

- `row_id_scores`
- `dynamic_columns`

对于 BM25：

- 若未请求 `score`，`dynamic_columns` 为空
- 若请求 `score`，返回 `Float64Array`

### 7.3 合并阶段

当前 `merge_search_index_entries()` 只合并 `row_id_scores`。改造后需同时合并动态列：

1. 校验 inline 和各 data-file 返回的动态列集合一致
2. 按现有逻辑合并 `row_id_scores`
3. 对每个动态列按同一顺序拼接 array
4. 生成一次全局排序索引
5. 对基础表 batch 和所有动态列统一执行 `take`
6. 如果存在 `limit`，同样只截一次

关键约束：排序索引只能生成一次，否则动态列和基础表行容易错位。

### 7.4 结果列顺序

`search` 最终结果列顺序：

1. `_row_id`
2. `projection` 选出的基础表列
3. `index_columns` 请求的动态列，按请求顺序追加

## 8. `TableScan` 执行方案

### 8.1 入口解析

在 [indexlake/src/table/scan.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/scan.rs) 中：

1. 先做现有的 filter 拆分
2. 计算 `index_filter_assignment`
3. 校验 `index_columns` 是否都能绑定到实际参与过滤的 index
4. 为每个参与 scan 的 index 构造各自的 `IndexResultOptions`

### 8.2 index filter 返回动态列

`index_scan_inline_rows()` 和 `filter_index_files_row_ids()` 当前只消费 `row_ids`。改造后：

- `Index::filter(filters, options)` 返回 `FilterIndexEntries`
- 其中 `dynamic_columns` 与 `row_ids` 一一对齐

示意：

```rust
pub struct FilterIndexEntries {
    pub row_ids: Vec<Uuid>,
    pub dynamic_columns: Vec<IndexResultColumn>,
}
```

### 8.3 多 index 交集与动态列合并

scan 场景比 search 复杂，因为可能有多个 index 同时参与过滤。

建议合并规则：

1. 每个 index 只返回自己负责的动态列
2. 先按现有逻辑对各 index 的 `row_ids` 求交集
3. 每个 index 的动态列仍保留“按本 index 的 row_ids 对齐”的原始形态
4. 表层把它们转换成按 `row_id` 查值的内部结构
5. 回表读取基础数据时，根据 `_row_id` 为每个 batch 补动态列

内部建议引入仅表层可见的辅助结构：

```rust
struct RowMappedDynamicColumn {
    field: FieldRef,
    row_ids: Vec<Uuid>,
    values: ArrayRef,
}
```

或者进一步转成：

```rust
struct DynamicColumnLookup {
    field: FieldRef,
    row_id_to_pos: HashMap<Uuid, usize>,
    values: ArrayRef,
}
```

这样在 batch 级别追加列时，可以按 `_row_id` 逐批 gather，而不需要一次性 materialize 整个 scan 结果。

### 8.4 流式追加动态列

scan 的关键要求是保留 streaming。

建议做法：

1. index 先产出候选 `row_id` 和动态列 lookup
2. 回表阶段仍按现在的 inline/data-file 分批读取基础行
3. 每个输出 batch 取出 `_row_id`
4. 对每个动态列按 `_row_id` 顺序 gather 出一个新 array
5. 把新 array 追加到当前 batch 后输出

这样可以保持：

- 不需要把 scan 结果全部读入内存
- 动态列和基础行天然同批次对齐

### 8.5 `offset/limit` 的处理

当前 index scan 路径没有复用 `TablePartitionScanner` 的窗口裁剪逻辑。扩展动态列时应一并修正。

建议方案：

1. 把 `offset/limit` 逻辑从 `TablePartitionScanner` 中抽成通用的 `LimitOffsetStream`
2. 普通 table scan 和 index scan 都在“基础列 + 动态列已经拼好之后”走同一层窗口裁剪

原因：

- 如果先裁剪基础 batch，再补动态列，会增加错位风险
- 统一在最终 batch 层裁剪，语义最稳定

## 9. Schema 与校验规则

### 9.1 输出 schema

`TableScan::output_schema()` 和 `TableSearch` 的最终输出 schema 都需要扩展：

1. 先生成基础表 schema
2. 再按请求顺序追加动态列 field

### 9.2 列名冲突

以下情况直接报错：

- 动态列输出名与基础表列重名
- 两个动态列输出名相同
- 同一 index 返回的实际 field 名与请求解析结果不一致

不建议自动覆盖或自动重命名，否则 schema 不可预测。

### 9.3 空结果

即使结果集为空，只要请求了动态列，也必须保留这些字段。

这样调用方才能稳定依赖 schema。

### 9.4 不支持的动态列

如果 index 不支持某个列名，应返回显式错误，例如：

```text
Unsupported result column `score` for index `btree_idx`
```

不应静默忽略。

## 10. 不同 index 的首期语义

### 10.1 BM25

- `search` 支持 `score: Float64`
- `scan` 暂时没有 filter 语义，因此当前没有可用动态列
- 如果未来 BM25 支持 `match_bm25(...)` 形式的 filter，则可直接复用 `filter + dynamic_columns` 机制

### 10.2 HNSW

- `search` 可自然支持 `distance` 或 `score`
- 因其当前只支持 search，不支持 filter，所以 scan 场景暂无动态列

### 10.3 BTree / RStar

- 当前主要用于 `scan` 过滤
- 首期可以先不实现具体动态列
- 但框架层必须允许它们未来返回例如 `matched_key`、`distance_to_boundary`、`spatial_relation` 等附加信息

结论是：框架层现在就统一支持 `scan + search` 两条路径，具体 index 先按能力逐个接入。

## 11. 对 DataFusion 的影响

当前 DataFusion 集成的 schema 来自固定的 `table.output_schema`，位于 [integrations/datafusion/src/table.rs](/C:/Users/lewis/workspace/indexlake/integrations/datafusion/src/table.rs)。

因此本期结论仍然是：

- Rust API 的 `TableScan` / `TableSearch` 先支持动态列
- DataFusion SQL 支持另开议题

否则 `TableProvider::schema()` 无法静态声明动态列。

## 12. 实现拆分建议

### 12.1 第一步：核心结构

改动文件：

- [indexlake/src/index/mod.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/index/mod.rs)

内容：

- 增加 `IndexColumnRequest`
- 增加 `RequestedIndexColumn`
- 增加 `IndexResultOptions`
- 扩展 `SearchIndexEntries`
- 扩展 `FilterIndexEntries`
- 修改 `Index::search()` / `Index::filter()` 签名

### 12.2 第二步：`TableSearch`

改动文件：

- [indexlake/src/table/mod.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/mod.rs)
- [indexlake/src/table/search.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/search.rs)

内容：

- `TableSearch` 增加 `index_columns`
- 入口校验和请求解析
- 搜索结果动态列合并
- 排序后追加动态列

### 12.3 第三步：`TableScan`

改动文件：

- [indexlake/src/table/scan.rs](/C:/Users/lewis/workspace/indexlake/indexlake/src/table/scan.rs)

内容：

- `TableScan` 增加 `index_columns`
- index filter 路径返回动态列
- 引入 batch 级动态列追加
- 抽出统一的 `offset/limit` 裁剪逻辑

### 12.4 第四步：index 逐个接入

改动文件：

- [indexes/bm25/src/index.rs](/C:/Users/lewis/workspace/indexlake/indexes/bm25/src/index.rs)
- [indexes/hnsw/src/index.rs](/C:/Users/lewis/workspace/indexlake/indexes/hnsw/src/index.rs)
- [indexes/btree/src/index.rs](/C:/Users/lewis/workspace/indexlake/indexes/btree/src/index.rs)
- [indexes/rstar/src/index.rs](/C:/Users/lewis/workspace/indexlake/indexes/rstar/src/index.rs)

内容：

- BM25 先支持 `search.score`
- 其他 index 先把接口接上，动态列可先返回空

### 12.5 第五步：测试

优先增加：

1. `search` 不请求动态列时行为不变
2. BM25 `search` 请求 `score` 时结果多一列
3. `scan` 请求动态列但 index 未参与时报错
4. `scan` 多 index 参与时列解析无歧义
5. `scan` 动态列在多批次输出下不乱序
6. `offset/limit` 在 index scan 路径与普通 scan 路径行为一致
7. 空结果仍保留动态列 schema

## 13. 风险与规避

### 13.1 动态列和基础行错位

规避：

- `search` 只生成一次排序索引
- `scan` 统一按 `_row_id` 做 batch 内 gather

### 13.2 多 index 请求歧义

规避：

- scan 入口要求列请求能唯一解析到实际参与的 index

### 13.3 空结果 schema 不稳定

规避：

- 动态列 field 在空 batch 也必须保留

### 13.4 `offset/limit` 行为分叉

规避：

- 把窗口裁剪抽成 scan 路径共用组件

## 14. 结论

该方案把“index 产生的附加信息”统一升级为结果集动态列，并同时覆盖：

- `TableSearch`：典型场景是 BM25/HNSW 返回 `score` / `distance`
- `TableScan`：典型场景是过滤型 index 在回表结果上追加 index 计算列

实现上最关键的点有三个：

1. `TableScan` 和 `TableSearch` 共用 `IndexColumnRequest`
2. `Index::search()` 和 `Index::filter()` 都能返回 `dynamic_columns`
3. scan 路径按 `_row_id` 流式补列，search 路径按统一排序索引补列

建议实现顺序：

1. 先落核心抽象
2. 再改 `TableSearch`
3. 再改 `TableScan` 并顺手修正 index scan 的 `offset/limit`
4. 最后让 BM25 先接入 `score`
