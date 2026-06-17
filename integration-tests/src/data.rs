use std::sync::Arc;

use arrow::array::{
    BinaryArray, FixedSizeListBuilder, Float32Builder, Int32Array, ListBuilder, RecordBatch,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use geo::{Geometry, Point};
use geozero::{CoordDimensions, ToWkb};
use indexlake::index::IndexKind;
use indexlake::storage::DataFileFormat;
use indexlake::table::{IndexCreation, Table, TableConfig, TableCreation, TableInsertion};
use indexlake::{Client, ILResult};
use indexlake_index_btree::{BTreeIndexKind, BTreeIndexParams};

use crate::utils::{assert_data_file_count, assert_inline_row_count};

pub async fn prepare_simple_testing_table(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
            Arc::new(Int32Array::from(vec![20, 21])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Charlie", "David"])),
            Arc::new(Int32Array::from(vec![22, 23])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    assert_inline_row_count(&table, |count| count > 0).await?;
    assert_data_file_count(&table, |count| count > 0).await?;

    Ok(table)
}

pub async fn prepare_simple_testing_table2(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
    ]));
    let table_config = TableConfig {
        preferred_data_file_format: format,
        ..Default::default()
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
            Arc::new(Int32Array::from(vec![20, 21, 22])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["David", "Eve", "Frank"])),
            Arc::new(Int32Array::from(vec![23, 24, 25])),
        ],
    )?;
    table
        .bypass_insert(Box::pin(futures::stream::iter(vec![Ok(record_batch)])))
        .await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(StringArray::from(vec!["George", "Hannah", "Ivy"])),
            Arc::new(Int32Array::from(vec![26, 27, 28])),
        ],
    )?;
    table
        .bypass_insert(Box::pin(futures::stream::iter(vec![Ok(record_batch)])))
        .await?;

    assert_inline_row_count(&table, |count| count == 3).await?;
    assert_data_file_count(&table, |count| count == 2).await?;

    Ok(table)
}

pub async fn prepare_simple_geom_table(client: &Client, format: DataFileFormat) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("geom", DataType::Binary, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(BinaryArray::from(vec![
                Geometry::from(Point::new(10.0, 10.0))
                    .to_wkb(CoordDimensions::xy())
                    .unwrap()
                    .as_slice(),
                Geometry::from(Point::new(11.0, 11.0))
                    .to_wkb(CoordDimensions::xy())
                    .unwrap()
                    .as_slice(),
            ])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(BinaryArray::from(vec![
                Geometry::from(Point::new(12.0, 12.0))
                    .to_wkb(CoordDimensions::xy())
                    .unwrap()
                    .as_slice(),
                Geometry::from(Point::new(13.0, 13.0))
                    .to_wkb(CoordDimensions::xy())
                    .unwrap()
                    .as_slice(),
            ])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    assert_inline_row_count(&table, |count| count > 0).await?;
    assert_data_file_count(&table, |count| count > 0).await?;

    Ok(table)
}

pub async fn prepare_btree_integer_table(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("integer", DataType::Int32, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(Int32Array::from(vec![100, 50, 75, 25, 90, 75])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    Ok(table)
}

pub async fn prepare_btree_string_table(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("string", DataType::Utf8, false),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6])),
            Arc::new(StringArray::from(vec![
                "apple",
                "banana",
                "cherry",
                "date",
                "elderberry",
                "cherry",
            ])),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    Ok(table)
}

pub async fn prepare_simple_fixed_size_vector_table(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let fixed_size_list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "vector",
            DataType::FixedSizeList(fixed_size_list_inner_field.clone(), 3),
            true,
        ),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let mut fixed_size_list_builder = FixedSizeListBuilder::new(
        Float32Builder::new(),
        3,
    )
    .with_field(fixed_size_list_inner_field.clone());
    fixed_size_list_builder.values().append_slice(&[10.0, 10.0, 10.0]);
    fixed_size_list_builder.append(true);
    fixed_size_list_builder.values().append_slice(&[20.0, 20.0, 20.0]);
    fixed_size_list_builder.append(true);
    fixed_size_list_builder.values().append_slice(&[30.0, 30.0, 30.0]);
    fixed_size_list_builder.append(true);
    fixed_size_list_builder.values().append_slice(&[40.0, 40.0, 40.0]);
    fixed_size_list_builder.append(true);
    let fixed_size_list_array = fixed_size_list_builder.finish();

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(fixed_size_list_array),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    assert_inline_row_count(&table, |count| count > 0).await?;
    assert_data_file_count(&table, |count| count > 0).await?;

    Ok(table)
}

pub async fn prepare_simple_vector_table(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", DataType::List(list_inner_field.clone()), true),
    ]));
    let table_config = TableConfig {
        inline_row_count_limit: 3,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;

    let table = client.load_table(&namespace_name, &table_name).await?;

    let mut list_builder =
        ListBuilder::new(Float32Builder::new()).with_field(list_inner_field.clone());
    list_builder.values().append_slice(&[10.0, 10.0, 10.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[20.0, 20.0, 20.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[30.0, 30.0, 30.0]);
    list_builder.append(true);
    list_builder.values().append_slice(&[40.0, 40.0, 40.0]);
    list_builder.append(true);
    let list_array = list_builder.finish();

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(list_array),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    assert_inline_row_count(&table, |count| count > 0).await?;
    assert_data_file_count(&table, |count| count > 0).await?;

    Ok(table)
}

pub async fn prepare_table_with_two_btree_indexes(
    client: &Client,
    format: DataFileFormat,
) -> ILResult<Table> {
    let namespace_name = uuid::Uuid::new_v4().to_string();
    client.create_namespace(&namespace_name, true).await?;

    let list_inner_field = Arc::new(Field::new("item", DataType::Float32, false));
    let table_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("age", DataType::Int32, false),
        Field::new_list(
            "vector",
            Arc::new(Field::new("item", DataType::Float32, false)),
            false,
        ),
    ]));
    // Keep rows inline so inline indexes are used
    let table_config = TableConfig {
        inline_row_count_limit: 100,
        parquet_row_group_size: 2,
        preferred_data_file_format: format,
    };
    let table_name = uuid::Uuid::new_v4().to_string();
    let table_creation = TableCreation {
        namespace_name: namespace_name.clone(),
        table_name: table_name.clone(),
        schema: table_schema.clone(),
        config: table_config,
        ..Default::default()
    };
    client.create_table(table_creation).await?;
    let table = client.load_table(&namespace_name, &table_name).await?;

    let mut vector_builder = ListBuilder::new(Float32Builder::new()).with_field(list_inner_field);
    for vector in [
        [10.0, 10.0, 10.0],
        [20.0, 20.0, 20.0],
        [30.0, 30.0, 30.0],
        [40.0, 40.0, 40.0],
    ] {
        vector_builder.values().append_slice(&vector);
        vector_builder.append(true);
    }

    let record_batch = RecordBatch::try_new(
        table_schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie", "David"])),
            Arc::new(Int32Array::from(vec![20, 21, 22, 23])),
            Arc::new(vector_builder.finish()),
        ],
    )?;
    table
        .insert(TableInsertion::new(vec![record_batch]))
        .await?;

    // Create btree index on name
    let name_index_creation = IndexCreation {
        name: "name_btree".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["name".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(name_index_creation).await?;

    // Reload table after create_index consumes self
    let table = client.load_table(&namespace_name, &table_name).await?;

    // Create btree index on age
    let age_index_creation = IndexCreation {
        name: "age_btree".to_string(),
        kind: BTreeIndexKind.kind().to_string(),
        key_columns: vec!["age".to_string()],
        params: Arc::new(BTreeIndexParams {}),
        concurrency: 1,
        if_not_exists: false,
    };
    table.create_index(age_index_creation).await?;

    // Reload table after create_index consumes self
    client.load_table(&namespace_name, &table_name).await
}
