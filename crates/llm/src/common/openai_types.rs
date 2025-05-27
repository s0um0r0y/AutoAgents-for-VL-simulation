use crate::llm::{ChatCompletionOptions, ChatMessage, ChatRole, TextGenerationOptions};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIStyleTextGenerationOptions {
    pub max_tokens: i32,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl From<TextGenerationOptions> for OpenAIStyleTextGenerationOptions {
    fn from(value: TextGenerationOptions) -> Self {
        Self {
            max_tokens: value.num_tokens,
            temperature: Some(0.7),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenAIStyleChatCompletionOptions {
    pub max_tokens: Option<i32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

impl From<ChatCompletionOptions> for OpenAIStyleChatCompletionOptions {
    fn from(value: ChatCompletionOptions) -> Self {
        Self {
            max_tokens: if value.num_tokens > 0 { Some(value.num_tokens) } else { None },
            temperature: Some(0.7),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStyleMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIStyleToolCall>>,
}

impl From<ChatMessage> for OpenAIStyleMessage {
    fn from(value: ChatMessage) -> Self {
        let role = match value.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        Self {
            role: role.to_string(),
            content: value.content,
            tool_calls: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIStyleTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIStyleFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIStyleFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStyleToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAIStyleToolCallFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIStyleToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OpenAIStyleChatCompletionRequest {
    pub model: String,
    pub messages: Vec<OpenAIStyleMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAIStyleTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
}

impl OpenAIStyleChatCompletionRequest {
    pub fn new(model: String) -> Self {
        Self {
            model,
            messages: vec![],
            tools: None,
            max_tokens: None,
            temperature: Some(0.7),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            stream: Some(false),
            format: None,
            keep_alive: None,
        }
    }

    pub fn set_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub fn set_messages(mut self, messages: Vec<OpenAIStyleMessage>) -> Self {
        self.messages = messages;
        self
    }

    pub fn set_tools(mut self, tools: Vec<OpenAIStyleTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn set_chat_options(mut self, options: OpenAIStyleChatCompletionOptions) -> Self {
        self.max_tokens = options.max_tokens;
        self.temperature = options.temperature;
        self.top_p = options.top_p;
        self.frequency_penalty = options.frequency_penalty;
        self.presence_penalty = options.presence_penalty;
        self
    }

    pub fn set_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    pub fn set_format(mut self, format: String) -> Self {
        self.format = Some(format);
        self
    }

    pub fn set_keep_alive(mut self, keep_alive: String) -> Self {
        self.keep_alive = Some(keep_alive);
        self
    }

    pub fn from_chat_messages(messages: Vec<ChatMessage>, model: String) -> Self {
        let openai_messages: Vec<OpenAIStyleMessage> = messages
            .into_iter()
            .map(OpenAIStyleMessage::from)
            .collect();

        Self::new(model).set_messages(openai_messages)
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleChatChoice {
    pub index: i32,
    pub message: OpenAIStyleResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleResponseMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAIStyleToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAIStyleChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OpenAIStyleToolCall>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleStreamChoice {
    pub index: i32,
    pub delta: OpenAIStyleStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIStyleStreamResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<OpenAIStyleStreamChoice>,
}

pub fn parse_openai_style_stream_chunk(chunk: &str) -> Option<String> {
    if let Some(data) = chunk.strip_prefix("data: ") {
        if data.trim() == "[DONE]" {
            return None;
        }
        Some(data.to_string())
    } else {
        None
    }
}