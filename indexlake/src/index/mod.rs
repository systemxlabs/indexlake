mod definition;
mod manager;

pub use definition::*;
pub use manager::*;
use uuid::Uuid;

use crate::ILResult;
use crate::expr::Expr;
use crate::storage::{InputFile, OutputFile};
use arrow::array::ArrayRef;
use arrow::array::RecordBatch;
use arrow::datatypes::FieldRef;
use std::any::Any;
use std::fmt::Debug;
use std::sync::Arc;

pub trait IndexKind: Debug + Send + Sync {
    // The kind of the index.
    fn kind(&self) -> &str;

    fn decode_params(&self, value: &str) -> ILResult<Arc<dyn IndexParams>>;

    fn supports(&self, index_def: &IndexDefinition) -> ILResult<()>;

    fn builder(&self, index_def: &IndexDefinitionRef) -> ILResult<Box<dyn IndexBuilder>>;

    fn supports_search(
        &self,
        index_def: &IndexDefinition,
        query: &dyn SearchQuery,
    ) -> ILResult<bool>;

    fn output_fields(
        &self,
        index_def: &IndexDefinition,
        columns: &[RequestedIndexColumn],
    ) -> ILResult<Vec<FieldRef>>;

    fn supports_filter(
        &self,
        index_def: &IndexDefinition,
        filter: &Expr,
    ) -> ILResult<FilterSupport>;
}

#[async_trait::async_trait]
pub trait IndexBuilder: Debug + Send + Sync {
    fn index_def(&self) -> &IndexDefinitionRef;

    fn append(&mut self, batch: &RecordBatch) -> ILResult<()>;

    async fn read_file(&mut self, input_file: Box<dyn InputFile>) -> ILResult<()>;

    async fn write_file(&mut self, output_file: Box<dyn OutputFile>) -> ILResult<()>;

    fn read_bytes(&mut self, buf: &[u8]) -> ILResult<()>;

    fn write_bytes(&mut self, buf: &mut Vec<u8>) -> ILResult<()>;

    fn build(&mut self) -> ILResult<Box<dyn Index>>;
}

#[async_trait::async_trait]
pub trait Index: Debug + Send + Sync {
    async fn search(
        &self,
        query: &dyn SearchQuery,
        options: &IndexResultOptions,
    ) -> ILResult<SearchIndexEntries>;

    async fn filter(
        &self,
        filters: &[Expr],
        options: &IndexResultOptions,
    ) -> ILResult<FilterIndexEntries>;
}

#[derive(Debug, Clone, Default)]
pub struct IndexResultOptions {
    pub columns: Vec<RequestedIndexColumn>,
}

#[derive(Debug, Clone)]
pub struct RequestedIndexColumn {
    pub name: String,
    pub output_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexColumnRequest {
    pub index_name: Option<String>,
    pub name: String,
    pub alias: Option<String>,
}

impl IndexColumnRequest {
    pub fn output_name(&self) -> &str {
        self.alias.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Debug, Clone)]
pub struct IndexResultColumn {
    pub field: FieldRef,
    pub values: ArrayRef,
}

#[derive(Debug, Clone)]
pub struct SearchIndexEntries {
    pub row_id_scores: Vec<RowIdScore>,
    pub score_higher_is_better: bool,
    pub dynamic_columns: Vec<IndexResultColumn>,
}

#[derive(Debug, Clone)]
pub struct RowIdScore {
    pub row_id: Uuid,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct FilterIndexEntries {
    pub row_ids: Vec<Uuid>,
    pub dynamic_columns: Vec<IndexResultColumn>,
}

#[derive(Debug, Clone, Copy)]
pub enum FilterSupport {
    Unsupported,
    Exact,
    Inexact,
}

pub trait SearchQuery: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn index_kind(&self) -> &str;

    fn limit(&self) -> Option<usize>;
}

pub trait IndexParams: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn encode(&self) -> ILResult<String>;
}
