use super::model::OllamaModel;
use crate::common::openai_types::{
    OpenAIStyleChatCompletionOptions, OpenAIStyleChatCompletionRequest, 
    OpenAIStyleTool
};
use crate::llm::{ChatCompletionOptions, TextGenerationOptions};
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

// Ollama-specific options that convert from OpenAI-style options
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

impl From<OllamaChatCompletionOptions> for OpenAIStyleChatCompletionOptions {
    fn from(value: OllamaChatCompletionOptions) -> Self {
        Self {
            max_tokens: Some(value.num_predict),
            temperature: Some(0.7),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
        }
    }
}

impl From<OpenAIStyleChatCompletionOptions> for OllamaChatCompletionOptions {
    fn from(value: OpenAIStyleChatCompletionOptions) -> Self {
        Self {
            num_predict: value.max_tokens.unwrap_or(1024),
        }
    }
}

// Use common OpenAI-style request but add Ollama-specific fields
pub type OllamaChatRequest = OpenAIStyleChatCompletionRequest;

// Ollama-specific tool type that maps to OpenAI-style
pub type OllamaTool = OpenAIStyleTool;

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

// Helper functions for Ollama-specific functionality
impl OllamaChatRequest {
    pub fn set_ollama_options(mut self, options: OpenAIStyleChatCompletionOptions) -> Self {
        // Since we're using the OpenAI-style request, we need to map num_predict to max_tokens
        self.max_tokens = options.max_tokens;
        self.temperature = options.temperature;
        self.top_p = options.top_p;
        self.frequency_penalty = options.frequency_penalty;
        self.presence_penalty = options.presence_penalty;
        self
    }

    pub fn with_ollama_defaults(model: String) -> Self {
        OpenAIStyleChatCompletionRequest::new(model)
            .set_keep_alive("5m".to_string())
    }
}
