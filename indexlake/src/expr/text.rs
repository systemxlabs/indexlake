use crate::expr::Expr;
use crate::{ILError, ILResult};

pub fn serialize_expr(expr: &Expr) -> ILResult<String> {
    serde_json::to_string(expr)
        .map_err(|e| ILError::internal(format!("Failed to serialize expr: {e:?}")))
}

pub fn deserialize_expr(text: &str) -> ILResult<Expr> {
    serde_json::from_str(text)
        .map_err(|e| ILError::invalid_input(format!("Failed to deserialize expr: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{col, lit};

    #[test]
    fn test_expr_json_roundtrip() {
        let expr = col("a").plus(lit(1i32)).eq(lit(2i32));
        let text = serialize_expr(&expr).unwrap();
        let parsed = deserialize_expr(&text).unwrap();
        assert_eq!(expr, parsed);
    }
}
