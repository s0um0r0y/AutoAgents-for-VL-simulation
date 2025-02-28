use crate::{error::LLMProviderError, Tool};
use async_trait::async_trait;
use futures::stream::BoxStream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt::Display;
use strum::EnumString;

#[async_trait]
pub trait LLM: Send + Sync {
    type Error: LLMProviderError + std::error::Error + Send + Sync + 'static;

    async fn text_generation(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> Result<TextGenerationResponse, Self::Error>;

    async fn text_generation_stream(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> BoxStream<'static, Result<TextGenerationResponse, Self::Error>>;

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error>;

    fn chat_completion_sync(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error>;

    async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> BoxStream<'static, Result<ChatCompletionResponse, Self::Error>>;

    fn name(&self) -> &str;

    fn model_name(&self) -> String;

    fn tools(&self) -> &Vec<Box<dyn Tool>>;

    fn register_tool(&mut self, tool: impl Tool + 'static);

    /// A helper method to call a tool by name with JSON arguments.
    fn call_tool(&self, tool_name: &str, args: Value) -> Option<Value> {
        for tool in self.tools() {
            if tool.name() == tool_name {
                return Some(tool.run(args));
            }
        }
        None
    }
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
    pub message: ChatResponseMessage,
    pub done: bool,
    #[serde(default)]
    pub done_reason: String,
    #[serde(default)]
    pub total_duration: i64,
    #[serde(default)]
    pub load_duration: i64,
    #[serde(default)]
    pub prompt_eval_count: i32,
    #[serde(default)]
    pub prompt_eval_duration: i64,
    #[serde(default)]
    pub eval_count: i32,
    #[serde(default)]
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
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    pub role: ChatRole,
    #[serde(default)]
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<ToolCallRequest>,
}
