use std::fmt::Debug;
use thiserror::Error;

pub trait LLMProviderError {}

#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Text generation error: {0}")]
    TextGeneration(String),

    #[error("Embedding generation error: {0}")]
    EmbeddingGeneration(String),

    #[error("Configuration error: {0}")]
    Configuration(String),
}
