mod builder;
mod distance;
mod graph;
mod index;
mod kind;
mod serialization;

pub use builder::*;
pub use index::*;
pub use kind::*;

/// Graph internals exposed for benchmarks and diagnostics.
///
/// This is not a stable API: the graph layout and method signatures may change
/// with the implementation.
#[doc(hidden)]
pub mod internals {
    pub use crate::graph::{HnswGraph, NodeId};
}
