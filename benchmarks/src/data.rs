use std::sync::Arc;

use arrow::array::*;
use arrow::datatypes::*;

pub fn arrow_table_schema() -> SchemaRef {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("content", DataType::Utf8, false),
        Field::new("data", DataType::Binary, true),
    ]);
    Arc::new(schema)
}

pub fn new_record_batch(num_rows: usize) -> RecordBatch {
    let schema = arrow_table_schema();
    let id_array = Int32Array::from_iter_values(0..num_rows as i32);
    let content = "content".repeat(1000);
    let content_array = StringArray::from_iter_values(vec![content; num_rows]);
    let data = b"data".repeat(1000);
    let data_array = BinaryArray::from_iter_values(vec![data; num_rows]);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(content_array) as ArrayRef,
            Arc::new(data_array) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn arrow_bm25_table_schema() -> SchemaRef {
    let schema = Schema::new(vec![
        Field::new("title", DataType::Utf8, false),
        Field::new("content", DataType::Utf8, false),
    ]);
    Arc::new(schema)
}

pub fn new_bm25_record_batch(num_rows: usize) -> RecordBatch {
    let schema = arrow_bm25_table_schema();
    let title_array = StringArray::from_iter_values(vec!["title"; num_rows]);

    let content = "如前所述，哈希分区可以帮助减少热点。但是，它不能完全避免它们：在极端情况下，所有的读写操作都是针对同一个键的，所有的请求都会被路由到同一个分区。
这种场景也许并不常见，但并非闻所未闻：例如，在社交媒体网站上，一个拥有数百万追随者的名人用户在做某事时可能会引发一场风暴【14】。这个事件可能导致同一个键的大量写入（键可能是名人的用户 ID，或者人们正在评论的动作的 ID）。哈希策略不起作用，因为两个相同 ID 的哈希值仍然是相同的。
如今，大多数数据系统无法自动补偿这种高度偏斜的负载，因此应用程序有责任减少偏斜。例如，如果一个主键被认为是非常火爆的，一个简单的方法是在主键的开始或结尾添加一个随机数。只要一个两位数的十进制随机数就可以将主键分散为 100 种不同的主键，从而存储在不同的分区中。
然而，将主键进行分割之后，任何读取都必须要做额外的工作，因为他们必须从所有 100 个主键分布中读取数据并将其合并。此技术还需要额外的记录：只需要对少量热点附加随机数；对于写入吞吐量低的绝大多数主键来说是不必要的开销。因此，你还需要一些方法来跟踪哪些键需要被分割。
也许在将来，数据系统将能够自动检测和补偿偏斜的工作负载；但现在，你需要自己来权衡。".to_string();

    let mut contents = vec![content; num_rows - 1];

    contents.push("这本书整整写了两年。两年里忧世伤生，屡想中止。由于杨绛女士不断的督促，替我挡了许多事，省出时间来，得以锱铢积累地写完。照例这本书该献给她。不过，近来觉得献书也像“致身于国”、“还政于民”等等佳话，只是语言幻成的空花泡影，名说交付出去，其实只仿佛魔术家玩的飞刀，放手而并没有脱手。随你怎样把作品奉献给人，作品总是作者自已的。大不了一本书，还不值得这样精巧地不老实，因此罢了。".to_string());

    let content_array = StringArray::from_iter_values(contents);

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(title_array) as ArrayRef,
            Arc::new(content_array) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn arrow_hnsw_vector_inner_field() -> Arc<Field> {
    Arc::new(Field::new("f32", DataType::Float32, false))
}

pub fn arrow_hnsw_table_schema() -> SchemaRef {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::List(arrow_hnsw_vector_inner_field()),
            false,
        ),
    ]);
    Arc::new(schema)
}

pub fn new_hnsw_record_batch(num_rows: usize) -> RecordBatch {
    let schema = arrow_hnsw_table_schema();
    let id_array = Int32Array::from_iter_values(0..num_rows as i32);

    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(arrow_hnsw_vector_inner_field());
    for _ in 0..num_rows {
        let rand = rand::random_range(0f32..1000f32);
        list_builder.values().append_slice(&[rand; 1024]);
        list_builder.append(true);
    }
    let vector_array = list_builder.finish();

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array) as ArrayRef,
            Arc::new(vector_array) as ArrayRef,
        ],
    )
    .unwrap()
}

pub fn arrow_btree_integer_table_schema() -> SchemaRef {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("integer", DataType::Int32, false),
    ]);
    Arc::new(schema)
}

pub fn new_btree_integer_record_batch(num_rows: usize) -> RecordBatch {
    let schema = arrow_btree_integer_table_schema();
    let id_array = Int32Array::from_iter_values(0..num_rows as i32);
    let integer_array =
        Int32Array::from_iter_values((0..num_rows).map(|_| rand::random_range(0..10000)));

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(integer_array)]).unwrap()
}

pub fn arrow_btree_string_table_schema() -> SchemaRef {
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("string", DataType::Utf8, false),
    ]);
    Arc::new(schema)
}

pub fn new_btree_string_record_batch(num_rows: usize) -> RecordBatch {
    let schema = arrow_btree_string_table_schema();
    let id_array = Int32Array::from_iter_values(0..num_rows as i32);
    let strings = [
        "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry",
    ];
    let string_array = StringArray::from_iter_values(
        (0..num_rows).map(|i| format!("{}{:06}", strings[i % strings.len()], i)),
    );

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(string_array)]).unwrap()
}
