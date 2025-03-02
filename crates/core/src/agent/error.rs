use std::fmt::Debug;
use thiserror::Error;

pub trait LLMProviderError {}

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("The Current Model {0} does not support tool calling")]
    ModelToolNotSupported(String),
}
