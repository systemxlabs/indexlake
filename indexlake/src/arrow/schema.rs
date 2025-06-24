use crate::{
    ILError, ILResult,
    record::{DataType, Field, Schema},
};
use arrow::datatypes::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

pub fn arrow_schema_to_schema(arrow_schema: &ArrowSchema) -> ILResult<Schema> {
    let mut fields = Vec::with_capacity(arrow_schema.fields.len());
    for arrow_field in arrow_schema.fields.iter() {
        let datatype = arrow_datatype_to_datatype(&arrow_field.data_type())?;
        fields.push(
            Field::new(
                arrow_field.name().clone(),
                datatype,
                arrow_field.is_nullable(),
            )
            .with_metadata(arrow_field.metadata().clone()),
        );
    }
    Ok(Schema::new_with_metadata(
        fields,
        arrow_schema.metadata.clone(),
    ))
}

pub fn schema_to_arrow_schema(schema: &Schema) -> ILResult<ArrowSchema> {
    let mut arrow_fields = Vec::with_capacity(schema.fields.len());
    for field in schema.fields.iter() {
        let arrow_datatype = datatype_to_arrow_datatype(&field.data_type);
        arrow_fields.push(
            ArrowField::new(field.name.clone(), arrow_datatype, field.nullable)
                .with_metadata(field.metadata.clone()),
        );
    }
    Ok(ArrowSchema::new_with_metadata(
        arrow_fields,
        schema.metadata.clone(),
    ))
}

pub fn arrow_datatype_to_datatype(datatype: &ArrowDataType) -> ILResult<DataType> {
    match datatype {
        ArrowDataType::Int16 => Ok(DataType::Int16),
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Boolean),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        ArrowDataType::Binary => Ok(DataType::Binary),
        _ => Err(ILError::InternalError(format!(
            "Unsupported arrow datatype: {:?}",
            datatype
        ))),
    }
}

pub fn datatype_to_arrow_datatype(datatype: &DataType) -> ArrowDataType {
    match datatype {
        DataType::Int16 => ArrowDataType::Int16,
        DataType::Int32 => ArrowDataType::Int32,
        DataType::Int64 => ArrowDataType::Int64,
        DataType::Float32 => ArrowDataType::Float32,
        DataType::Float64 => ArrowDataType::Float64,
        DataType::Boolean => ArrowDataType::Boolean,
        DataType::Utf8 => ArrowDataType::Utf8,
        DataType::Binary => ArrowDataType::Binary,
    }
}
