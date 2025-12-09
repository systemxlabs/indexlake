mod builder;
mod catalog;

pub use builder::*;
pub use catalog::*;

pub struct SqlDisplay<'a>(pub &'a str);

impl<'a> std::fmt::Display for SqlDisplay<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.starts_with("INSERT") && self.0.len() > 200 {
            write!(f, "{}", self.0.chars().take(200).collect::<String>())
        } else {
            write!(f, "{}", self.0)
        }
    }
}
