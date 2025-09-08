use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    BinaryArray, Float32Builder, Int32Array, ListBuilder, RecordBatch, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use geo::{Geometry, Point};
use geozero::{CoordDimensions, ToWkb};
use indexlake::storage::DataFileFormat;
use indexlake::table::{Table, TableConfig, TableCreation, TableInsertion};
use indexlake::{Client, ILResult};

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
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
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
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
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
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
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
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
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
        default_values: HashMap::new(),
        config: table_config,
        if_not_exists: false,
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
        .insert(TableInsertion::new(vec![record_batch]).with_force_inline(true))
        .await?;

    // wait for dump task to finish
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    Ok(table)
}
