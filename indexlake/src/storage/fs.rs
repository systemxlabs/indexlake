use std::path::PathBuf;

use opendal::Operator;
use opendal::layers::RetryLayer;
use opendal::services::FsConfig;

use crate::ILResult;

#[derive(Debug, Clone)]
pub struct FsStorage {
    pub root: PathBuf,
}

impl FsStorage {
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    pub fn new_operator(&self) -> ILResult<Operator> {
        let mut cfg = FsConfig::default();
        cfg.root = Some(self.root.to_string_lossy().to_string());
        Ok(Operator::from_config(cfg)?
            .layer(RetryLayer::new())
            .finish())
    }
}
