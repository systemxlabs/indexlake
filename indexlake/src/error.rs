use std::panic::Location;

pub type ILResult<T> = Result<T, ILError>;

#[derive(Debug)]
pub enum ILError {
    InternalError(String, &'static Location<'static>),
    NotSupported(String, &'static Location<'static>),
    CatalogError(String, &'static Location<'static>),
    StorageError(String, &'static Location<'static>),
    IndexError(String, &'static Location<'static>),
    InvalidInput(String, &'static Location<'static>),
}

impl ILError {
    #[track_caller]
    pub fn internal(msg: impl Into<String>) -> Self {
        ILError::InternalError(msg.into(), Location::caller())
    }

    #[track_caller]
    pub fn not_supported(msg: impl Into<String>) -> Self {
        ILError::NotSupported(msg.into(), Location::caller())
    }

    #[track_caller]
    pub fn catalog(msg: impl Into<String>) -> Self {
        ILError::CatalogError(msg.into(), Location::caller())
    }

    #[track_caller]
    pub fn storage(msg: impl Into<String>) -> Self {
        ILError::StorageError(msg.into(), Location::caller())
    }

    #[track_caller]
    pub fn index(msg: impl Into<String>) -> Self {
        ILError::IndexError(msg.into(), Location::caller())
    }

    #[track_caller]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        ILError::InvalidInput(msg.into(), Location::caller())
    }
}

impl std::fmt::Display for ILError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ILError::InternalError(msg, loc) => write!(f, "Internal error: {msg} at {loc}"),
            ILError::NotSupported(msg, loc) => write!(f, "Not supported: {msg} at {loc}"),
            ILError::CatalogError(msg, loc) => write!(f, "Catalog error: {msg} at {loc}"),
            ILError::StorageError(msg, loc) => write!(f, "Storage error: {msg} at {loc}"),
            ILError::IndexError(msg, loc) => write!(f, "Index error: {msg} at {loc}"),
            ILError::InvalidInput(msg, loc) => write!(f, "Invalid input: {msg} at {loc}"),
        }
    }
}

impl std::error::Error for ILError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

impl From<opendal::Error> for ILError {
    #[track_caller]
    fn from(err: opendal::Error) -> Self {
        ILError::StorageError(err.to_string(), Location::caller())
    }
}

impl From<parquet::errors::ParquetError> for ILError {
    #[track_caller]
    fn from(err: parquet::errors::ParquetError) -> Self {
        ILError::StorageError(err.to_string(), Location::caller())
    }
}

impl From<arrow::error::ArrowError> for ILError {
    #[track_caller]
    fn from(err: arrow::error::ArrowError) -> Self {
        ILError::InternalError(err.to_string(), Location::caller())
    }
}

impl From<tokio::task::JoinError> for ILError {
    #[track_caller]
    fn from(err: tokio::task::JoinError) -> Self {
        ILError::InternalError(format!("Tokio join error: {err}"), Location::caller())
    }
}

impl From<uuid::Error> for ILError {
    #[track_caller]
    fn from(err: uuid::Error) -> Self {
        ILError::InternalError(format!("UUID error: {err}"), Location::caller())
    }
}
