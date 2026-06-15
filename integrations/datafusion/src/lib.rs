mod codec;
mod delete;
mod expr;
mod insert;
mod protobuf;
mod scan;
mod search;
mod table;
mod update;

pub use codec::*;
pub use delete::*;
pub use expr::*;
pub use insert::*;
pub(crate) use protobuf::*;
pub use scan::*;
pub use search::*;
pub use table::*;
pub use update::*;

use arrow::datatypes::Schema;

pub(crate) fn schema_projection_equals(left: &Schema, right: &Schema) -> bool {
    if left.fields.len() != right.fields.len() {
        return false;
    }
    for (left_field, right_field) in left.fields.iter().zip(right.fields.iter()) {
        if left_field.name() != right_field.name() {
            return false;
        }
    }
    true
}
