use std::collections::HashMap;

use derive_visitor::{Drive, Visitor, VisitorMut};
use uuid::Uuid;

use crate::{catalog::INTERNAL_ROW_ID_FIELD_NAME, expr::Expr};

#[derive(Debug, Default, Visitor)]
#[visitor(Expr(enter))]
pub struct ColumnRecorder {
    columns: Vec<String>,
}

impl ColumnRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enter_expr(&mut self, expr: &Expr) {
        if let Expr::Column(name) = expr {
            self.columns.push(name.clone());
        }
    }
}

pub fn visited_columns(expr: &Expr) -> Vec<String> {
    let mut recorder = ColumnRecorder::new();
    expr.drive(&mut recorder);
    recorder.columns
}

#[derive(Debug, VisitorMut)]
#[visitor(Expr(enter))]
pub struct ColumnReplacer<'a> {
    pub field_name_id_map: &'a HashMap<String, Uuid>,
    pub fail_col: Option<String>,
}

impl<'a> ColumnReplacer<'a> {
    pub fn new(field_name_id_map: &'a HashMap<String, Uuid>) -> Self {
        Self {
            field_name_id_map,
            fail_col: None,
        }
    }

    pub fn enter_expr(&mut self, expr: &mut Expr) {
        if let Expr::Column(name) = expr
            && name != INTERNAL_ROW_ID_FIELD_NAME
        {
            if let Some(field_id) = self.field_name_id_map.get(name) {
                *name = hex::encode(field_id)
            } else {
                self.fail_col = Some(name.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::expr::{col, lit};

    use super::*;

    #[test]
    fn test_visited_columns() {
        let expr = col("a").eq(col("b").plus(lit(1)));
        let columns = visited_columns(&expr);
        assert_eq!(columns, vec!["a", "b"]);
    }
}
