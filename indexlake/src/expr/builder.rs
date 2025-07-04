use crate::{
    catalog::Scalar,
    expr::{BinaryExpr, BinaryOp, Expr, like::LikeExpr},
};

impl Expr {
    pub fn eq(self, other: Expr) -> Expr {
        Expr::BinaryExpr(BinaryExpr {
            left: Box::new(self),
            op: BinaryOp::Eq,
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
        Expr::LikeExpr(LikeExpr::new(false, Box::new(self), Box::new(other), false))
    }

    /// Return `self NOT LIKE other`
    pub fn not_like(self, other: Expr) -> Expr {
        Expr::LikeExpr(LikeExpr::new(true, Box::new(self), Box::new(other), false))
    }

    /// Return `self ILIKE other`
    pub fn ilike(self, other: Expr) -> Expr {
        Expr::LikeExpr(LikeExpr::new(false, Box::new(self), Box::new(other), true))
    }

    /// Return `self NOT ILIKE other`
    pub fn not_ilike(self, other: Expr) -> Expr {
        Expr::LikeExpr(LikeExpr::new(true, Box::new(self), Box::new(other), true))
    }
}

pub fn col(name: &str) -> Expr {
    Expr::Column(name.to_string())
}

pub fn lit(value: impl Into<Scalar>) -> Expr {
    Expr::Literal(value.into())
}
