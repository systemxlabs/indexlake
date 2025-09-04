use arrow::compute::can_cast_types;
use arrow::datatypes::DataType;
use arrow_schema::Schema;

use crate::catalog::Scalar;
use crate::expr::like::Like;
use crate::expr::{BinaryExpr, BinaryOp, Case, Cast, Expr, Function, Literal, TryCast};
use crate::{ILError, ILResult};

impl Expr {
    pub fn eq(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::Eq,
            right: Box::new(other),
        })
    }

    pub fn neq(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::NotEq,
            right: Box::new(other),
        })
    }

    pub fn gt(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::Gt,
            right: Box::new(other),
        })
    }

    pub fn gteq(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::GtEq,
            right: Box::new(other),
        })
    }

    pub fn lt(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::Lt,
            right: Box::new(other),
        })
    }

    pub fn lteq(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::LtEq,
            right: Box::new(other),
        })
    }

    pub fn plus(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::Plus,
            right: Box::new(other),
        })
    }

    pub fn and(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::And,
            right: Box::new(other),
        })
    }

    pub fn is_null(self) -> Expr {
        Expr::IsNull(Box::new(self))
    }

    pub fn is_not_null(self) -> Expr {
        Expr::IsNotNull(Box::new(self))
    }

    /// Return `self LIKE other`
    pub fn like(self, other: Expr) -> Expr {
        Expr::Like(Like::new(false, Box::new(self), Box::new(other), false))
    }

    /// Return `self NOT LIKE other`
    pub fn not_like(self, other: Expr) -> Expr {
        Expr::Like(Like::new(true, Box::new(self), Box::new(other), false))
    }

    /// Return `self ILIKE other`
    pub fn ilike(self, other: Expr) -> Expr {
        Expr::Like(Like::new(false, Box::new(self), Box::new(other), true))
    }

    /// Return `self NOT ILIKE other`
    pub fn not_ilike(self, other: Expr) -> Expr {
        Expr::Like(Like::new(true, Box::new(self), Box::new(other), true))
    }
}

pub fn col(name: &str) -> Expr {
    Expr::Column(name.to_string())
}

pub fn lit(value: impl Into<Scalar>) -> Expr {
    Expr::Literal(Literal {
        value: value.into(),
    })
}

pub fn func(name: impl Into<String>, args: Vec<Expr>, return_type: DataType) -> Expr {
    Expr::Function(Function {
        name: name.into(),
        args,
        return_type,
    })
}

pub fn try_cast(expr: Expr, schema: &Schema, cast_type: DataType) -> ILResult<Expr> {
    let expr_type = expr.data_type(schema)?;
    if expr_type == cast_type {
        Ok(expr)
    } else if can_cast_types(&expr_type, &cast_type) {
        Ok(Expr::TryCast(TryCast {
            expr: Box::new(expr),
            cast_type,
        }))
    } else {
        Err(ILError::not_supported(format!(
            "Unsupported TRY_CAST from {expr_type} to {cast_type}"
        )))
    }
}

impl From<Scalar> for Expr {
    fn from(value: Scalar) -> Self {
        Expr::Literal(Literal { value })
    }
}

impl From<BinaryExpr> for Expr {
    fn from(value: BinaryExpr) -> Self {
        Expr::BinaryExpr(value)
    }
}

impl From<Cast> for Expr {
    fn from(value: Cast) -> Self {
        Expr::Cast(value)
    }
}

impl From<TryCast> for Expr {
    fn from(value: TryCast) -> Self {
        Expr::TryCast(value)
    }
}

impl From<Like> for Expr {
    fn from(value: Like) -> Self {
        Expr::Like(value)
    }
}

impl From<Case> for Expr {
    fn from(value: Case) -> Self {
        Expr::Case(value)
    }
}
