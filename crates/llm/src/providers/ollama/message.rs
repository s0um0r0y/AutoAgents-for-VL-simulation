use super::model::OllamaModel;
use crate::llm::{ChatCompletionOptions, ChatMessage, ChatRole, TextGenerationOptions};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaTextGenerationOptions {
    num_predict: i32,
}

impl From<TextGenerationOptions> for OllamaTextGenerationOptions {
    fn from(value: TextGenerationOptions) -> Self {
        Self {
            num_predict: value.num_tokens,
        }
    }
}

macro_rules! common_setters_for_request {
    () => {
        pub fn set_model(mut self, model: String) -> Self {
            self.model = model;
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
    };
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaGenerateRequest {
    pub model: String,
    pub prompt: String,
    pub images: Option<Vec<String>>,
    pub format: Option<String>,
    pub stream: Option<bool>,
    pub system: Option<String>,
    pub keep_alive: Option<String>,
    pub options: Option<OllamaTextGenerationOptions>,
}

impl Default for OllamaGenerateRequest {
    fn default() -> Self {
        Self {
            model: OllamaModel::DeepSeekR18B.to_string(),
            prompt: "".to_string(),
            stream: Some(false),
            format: None,
            images: None,
            system: Some("You are a helpful assistant".to_string()),
            keep_alive: Some("5m".to_string()),
            options: Some(OllamaTextGenerationOptions::default()),
        }
    }
}

#[allow(dead_code)]
impl OllamaGenerateRequest {
    common_setters_for_request!();

    pub fn set_options(mut self, options: OllamaTextGenerationOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn set_system_prompt(mut self, system_prompt: String) -> Self {
        self.system = Some(system_prompt);
        self
    }

    pub fn set_prompt(mut self, prompt: String) -> Self {
        self.prompt = prompt;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OllamaChatCompletionOptions {
    num_predict: i32,
}

impl From<ChatCompletionOptions> for OllamaChatCompletionOptions {
    fn from(value: ChatCompletionOptions) -> Self {
        Self {
            num_predict: value.num_tokens,
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaChatMessage {
    pub role: ChatRole,
    pub content: String,
    pub images: Option<Vec<String>>,
    pub tool_calls: Option<Vec<String>>,
}

impl From<ChatMessage> for OllamaChatMessage {
    fn from(value: ChatMessage) -> Self {
        Self {
            role: value.role,
            content: value.content,
            images: None,
            tool_calls: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct OllamaChatMessages(pub Vec<OllamaChatMessage>);

impl From<Vec<ChatMessage>> for OllamaChatMessages {
    fn from(value: Vec<ChatMessage>) -> Self {
        let messages = value.into_iter().map(OllamaChatMessage::from).collect();
        OllamaChatMessages(messages)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaTool {
    #[serde(rename = "type")]
    pub _type: String,
    pub function: OllamaFunction,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // You can further define a struct if you want a stricter schema.
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: OllamaChatMessages,
    pub tools: Option<Vec<OllamaTool>>,
    pub format: Option<String>,
    pub stream: Option<bool>,
    pub keep_alive: Option<String>,
    pub options: Option<OllamaChatCompletionOptions>,
}

impl Default for OllamaChatRequest {
    fn default() -> Self {
        Self {
            model: OllamaModel::DeepSeekR18B.to_string(),
            messages: OllamaChatMessages::default(),
            tools: None,
            stream: Some(false),
            format: None,
            keep_alive: Some("5m".to_string()),
            options: Some(OllamaChatCompletionOptions::default()),
        }
    }
}

#[allow(dead_code)]
impl OllamaChatRequest {
    common_setters_for_request!();

    pub fn set_messages<A: Into<OllamaChatMessages>>(mut self, messages: A) -> Self {
        self.messages = messages.into();
        self
    }

    pub fn set_tools(mut self, tools: Vec<OllamaTool>) -> Self {
        self.tools = Some(tools);
        self
    }

    pub fn set_options(mut self, options: OllamaChatCompletionOptions) -> Self {
        self.options = Some(options);
        self
    }

    pub fn from_chat_messages(messages: Vec<ChatMessage>) -> Self {
        let ollama_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .cloned()
            .map(OllamaChatMessage::from)
            .collect();

        Self {
            messages: OllamaChatMessages(ollama_messages),
            tools: None,
            ..Default::default()
        }
    }
}
