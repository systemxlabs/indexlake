use opendal::layers::RetryLayer;
use opendal::services::S3Config;
use opendal::{Configurator, Operator};

use crate::ILResult;

#[derive(Debug, Clone)]
pub struct S3Storage {
    pub config: S3Config,
    pub bucket: String,
}

impl S3Storage {
    pub fn new(config: S3Config, bucket: String) -> Self {
        Self { config, bucket }
    }

    pub fn new_operator(&self) -> ILResult<Operator> {
        let builder = self.config.clone().into_builder().bucket(&self.bucket);
        Ok(Operator::new(builder)?.layer(RetryLayer::new()).finish())
    }
}
