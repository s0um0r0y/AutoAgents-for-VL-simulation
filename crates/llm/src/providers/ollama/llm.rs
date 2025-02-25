use super::{
    api::OllamaAPI,
    message::{OllamaGenerateRequest, OllamaTextGenerationOptions},
    model::OllamaModel,
};
use crate::{
    error::{LLMError, LLMProviderError},
    llm::{
        ChatCompletionOptions, ChatCompletionResponse, ChatMessage, TextGenerationOptions,
        TextGenerationResponse, LLM,
    },
    net::http_request::HTTPRequest,
    providers::ollama::message::{OllamaChatCompletionOptions, OllamaChatRequest},
    utils,
};
use async_trait::async_trait;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OllamaError {
    #[error("API error: {0}")]
    Api(String),
    #[error("Parsing error: {0}")]
    Parsing(String),
    #[error("LLM Error: {0}")]
    GenericLLMError(#[from] LLMError),
}

impl LLMProviderError for OllamaError {}

#[derive(Debug, Clone)]
pub struct Ollama<'a> {
    base_url: &'a str,
    pub model: Option<OllamaModel>,
}

impl Default for Ollama<'_> {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434",
            model: Some(OllamaModel::DeepSeekR18B),
        }
    }
}

impl<'a> Ollama<'a> {
    pub fn new() -> Self {
        Default::default()
    }

    fn model(&self) -> Result<String, LLMError> {
        let model = self.model.clone();
        model
            .map(|s| s.to_string())
            .ok_or_else(|| LLMError::Configuration("Model not set".to_string()))
    }

    fn base_url(&self) -> Result<String, LLMError> {
        Ok(self.base_url.to_string())
    }

    pub fn with_base_url(mut self, base_url: &'a str) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_model(mut self, model: OllamaModel) -> Self {
        self.model = Some(model);
        self
    }
}

#[async_trait]
impl<'a> LLM for Ollama<'a> {
    type Error = OllamaError;

    fn name(&self) -> &'a str {
        "Ollama"
    }

    async fn text_generation(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> Result<TextGenerationResponse, Self::Error> {
        let model = self.model()?;
        let base_url = self.base_url()?;
        let options: OllamaTextGenerationOptions = options.unwrap_or_default().into();
        let body = OllamaGenerateRequest::default()
            .set_model(model)
            .set_options(options)
            .set_system_prompt(system_prompt.into())
            .set_prompt(prompt.into());

        let url = utils::create_model_url(base_url, OllamaAPI::TextGenerate);
        let text = HTTPRequest::request(url, serde_json::json!(body))
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        let ollama_response: TextGenerationResponse =
            serde_json::from_str(&text).map_err(|e| OllamaError::Parsing(e.to_string()))?;

        Ok(ollama_response)
    }

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error> {
        let model = self.model()?;
        let base_url = self.base_url()?;
        let options: OllamaChatCompletionOptions = options.unwrap_or_default().into();
        let body = OllamaChatRequest::default()
            .set_model(model)
            .set_messages(messages)
            .set_options(options);

        let url = utils::create_model_url(base_url, OllamaAPI::ChatCompletion);
        let text = HTTPRequest::request(url, serde_json::json!(body))
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        let ollama_response: ChatCompletionResponse =
            serde_json::from_str(&text).map_err(|e| OllamaError::Parsing(e.to_string()))?;

        Ok(ollama_response)
    }

    fn model_name(&self) -> String {
        match &self.model {
            Some(model) => model.to_string(),
            None => "".to_string(),
        }
    }
}
