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
use futures::stream::TryStreamExt;
use futures::stream::{self, BoxStream, StreamExt};
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

    pub async fn text_generation_stream(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> BoxStream<'static, Result<TextGenerationResponse, OllamaError>> {
        // Validate configuration; if there's an error, return a stream that immediately yields it.
        let model = match self.model() {
            Ok(m) => m,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };
        let base_url = match self.base_url() {
            Ok(url) => url,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };

        let options: OllamaTextGenerationOptions = options.unwrap_or_default().into();
        let body = OllamaGenerateRequest::default()
            .set_model(model)
            .set_options(options)
            .set_stream(true)
            .set_system_prompt(system_prompt.into())
            .set_prompt(prompt.into());

        let url = utils::create_model_url(base_url, OllamaAPI::TextGenerate);

        // Create a future that resolves to the inner stream.
        let stream_future = HTTPRequest::stream_request(url, serde_json::json!(body));

        stream::once(stream_future)
            .then(|result| async move {
                match result {
                    // On success, convert inner errors from `reqwest::Error` to `OllamaError`
                    Ok(inner_stream) => inner_stream
                        .map_err(|e| OllamaError::Api(e.to_string()))
                        .boxed(),
                    // On error, create a stream that yields a single error of type `OllamaError`
                    Err(e) => stream::iter(vec![Err(OllamaError::Api(e.to_string()))]).boxed(),
                }
            })
            .flatten()
            .map(|chunk_result| match chunk_result {
                Ok(chunk) => {
                    serde_json::from_str(&chunk).map_err(|e| OllamaError::Parsing(e.to_string()))
                }
                // Here, the error is already an `OllamaError`
                Err(e) => Err(e),
            })
            .boxed()
    }

    pub async fn chat_completion_stream(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> BoxStream<'static, Result<ChatCompletionResponse, OllamaError>> {
        let model = match self.model() {
            Ok(m) => m,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };
        let base_url = match self.base_url() {
            Ok(url) => url,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };

        let options: OllamaChatCompletionOptions = options.unwrap_or_default().into();
        let body = OllamaChatRequest::default()
            .set_model(model)
            .set_messages(messages)
            .set_stream(true)
            .set_options(options);

        let url = utils::create_model_url(base_url, OllamaAPI::ChatCompletion);

        let stream_future = HTTPRequest::stream_request(url, serde_json::json!(body));

        stream::once(stream_future)
            .then(|result| async move {
                match result {
                    // On success, convert inner errors from `reqwest::Error` to `OllamaError`
                    Ok(inner_stream) => inner_stream
                        .map_err(|e| OllamaError::Api(e.to_string()))
                        .boxed(),
                    // On error, create a stream that yields a single error of type `OllamaError`
                    Err(e) => stream::iter(vec![Err(OllamaError::Api(e.to_string()))]).boxed(),
                }
            })
            .flatten()
            .map(|chunk_result| match chunk_result {
                Ok(chunk) => {
                    serde_json::from_str(&chunk).map_err(|e| OllamaError::Parsing(e.to_string()))
                }
                // Here, the error is already an `OllamaError`
                Err(e) => Err(e),
            })
            .boxed()
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
