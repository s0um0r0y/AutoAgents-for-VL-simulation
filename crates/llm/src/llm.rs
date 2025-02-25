use crate::error::LLMProviderError;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Display;
use strum::EnumString;

#[async_trait]
pub trait LLM {
    type Error: LLMProviderError + std::error::Error + Send + Sync + 'static;

    async fn text_generation(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> Result<TextGenerationResponse, Self::Error>;

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error>;

    fn name(&self) -> &str;

    fn model_name(&self) -> String;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationOptions {
    pub num_tokens: i32,
}

impl Default for TextGenerationOptions {
    fn default() -> Self {
        Self { num_tokens: 1024 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenerationResponse {
    pub response: String,
}

impl Display for TextGenerationResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.response)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionOptions {
    pub num_tokens: i32,
}

impl Default for ChatCompletionOptions {
    fn default() -> Self {
        Self { num_tokens: -1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub model: String,
    pub created_at: String,
    pub message: ChatMessage,
    pub done: bool,
    pub done_reason: String,
    pub total_duration: i64,
    pub load_duration: i64,
    pub prompt_eval_count: i32,
    pub prompt_eval_duration: i64,
    pub eval_count: i32,
    pub eval_duration: i64,
}

impl Display for ChatCompletionResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.message)
    }
}

#[derive(Debug, Serialize, Deserialize, EnumString, Clone)]
pub enum ChatRole {
    #[strum(serialize = "system")]
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    #[strum(serialize = "user")]
    User,
    #[serde(rename = "assistant")]
    #[strum(serialize = "assistant")]
    Assistant,
    #[serde(rename = "tool")]
    #[strum(serialize = "tool")]
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}
