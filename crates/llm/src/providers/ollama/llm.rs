use super::{
    api::OllamaAPI,
    message::{
        OllamaChatCompletionOptions, OllamaChatRequest, OllamaGenerateRequest,
        OllamaTextGenerationOptions, OllamaTool,
    },
    model::OllamaModel,
};
use crate::{
    common::openai_types::{OpenAIStyleChatCompletionResponse, OpenAIStyleFunction},
    error::{LLMError, LLMProviderError},
    llm::{
        ChatCompletionOptions, ChatCompletionResponse, ChatMessage, ChatResponseMessage, ChatRole,
        TextGenerationOptions, TextGenerationResponse, ToolCallFunction, ToolCallRequest, LLM,
    },
    net::http_request::HTTPRequest,
    tool::Tool,
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

#[derive(Debug)]
pub struct Ollama {
    base_url: String,
    pub model: Option<OllamaModel>,
    pub tools: Vec<Box<dyn Tool>>,
}

impl Clone for Ollama {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            tools: vec![], // Tools cannot be cloned, so we start with an empty vec
        }
    }
}

impl Default for Ollama {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".into(),
            model: Some(OllamaModel::DeepSeekR18B),
            tools: vec![],
        }
    }
}

impl Ollama {
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

    pub fn set_base_url<T: Into<String>>(mut self, base_url: T) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn set_model(mut self, model: OllamaModel) -> Self {
        self.model = Some(model);
        self
    }
}

#[async_trait]
impl LLM for Ollama {
    type Error = OllamaError;

    fn name(&self) -> &'static str {
        "Ollama"
    }

    fn supports_tools(&self) -> bool {
        if let Some(ref model) = self.model {
            return model.supports_tools();
        }
        false
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
        let text = HTTPRequest::request(&url, serde_json::json!(body))
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        let ollama_response: TextGenerationResponse =
            serde_json::from_str(&text).map_err(|e| OllamaError::Parsing(e.to_string()))?;

        Ok(ollama_response)
    }

    async fn text_generation_stream(
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

    async fn chat_completion(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error> {
        let model = self.model()?;
        let base_url = self.base_url()?;
        let options: OllamaChatCompletionOptions = options.unwrap_or_default().into();
        let mut body = OllamaChatRequest::from_chat_messages(messages, model)
            .set_ollama_options(options.into());

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OllamaTool> = self
                .tools
                .iter()
                .map(|tool| OllamaTool {
                    tool_type: "function".into(),
                    function: OpenAIStyleFunction {
                        name: tool.name().to_string(),
                        description: tool.description().to_string(),
                        parameters: tool.args_schema(),
                    },
                })
                .collect();
            body = body.set_tools(tools);
        }

        let url = utils::create_model_url(base_url, OllamaAPI::ChatCompletion);
        let text = HTTPRequest::request(&url, serde_json::json!(body))
            .await
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        let ollama_response: OpenAIStyleChatCompletionResponse =
            serde_json::from_str(&text).map_err(|e| OllamaError::Parsing(e.to_string()))?;

        // Convert OpenAI-style response to our common format
        let choice = ollama_response
            .choices
            .first()
            .ok_or_else(|| OllamaError::Parsing("No choices in response".to_string()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .iter()
            .map(|tc| ToolCallRequest {
                function: ToolCallFunction {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                },
            })
            .collect();

        let response = ChatCompletionResponse {
            model: ollama_response.model,
            created_at: ollama_response.created.to_string(),
            message: ChatResponseMessage {
                role: match choice.message.role.as_str() {
                    "system" => ChatRole::System,
                    "user" => ChatRole::User,
                    "assistant" => ChatRole::Assistant,
                    "tool" => ChatRole::Tool,
                    _ => ChatRole::Assistant,
                },
                content: choice.message.content.clone().unwrap_or_default(),
                tool_calls,
            },
            done: choice.finish_reason.is_some(),
            done_reason: choice.finish_reason.clone().unwrap_or_default(),
            total_duration: 0,
            load_duration: 0,
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
            eval_count: 0,
            eval_duration: 0,
        };

        Ok(response)
    }

    fn chat_completion_sync(
        &self,
        messages: Vec<ChatMessage>,
        options: Option<ChatCompletionOptions>,
    ) -> Result<ChatCompletionResponse, Self::Error> {
        let model = self.model()?;
        let base_url = self.base_url()?;
        let options: OllamaChatCompletionOptions = options.unwrap_or_default().into();
        let mut body = OllamaChatRequest::from_chat_messages(messages, model)
            .set_ollama_options(options.into());

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OllamaTool> = self
                .tools
                .iter()
                .map(|tool| OllamaTool {
                    tool_type: "function".into(),
                    function: OpenAIStyleFunction {
                        name: tool.name().to_string(),
                        description: tool.description().to_string(),
                        parameters: tool.args_schema(),
                    },
                })
                .collect();
            body = body.set_tools(tools);
        }

        let url = utils::create_model_url(base_url, OllamaAPI::ChatCompletion);
        let text = HTTPRequest::request_sync(&url, serde_json::json!(body))
            .map_err(|e| OllamaError::Api(e.to_string()))?;

        let ollama_response: OpenAIStyleChatCompletionResponse =
            serde_json::from_str(&text).map_err(|e| OllamaError::Parsing(e.to_string()))?;

        // Convert OpenAI-style response to our common format
        let choice = ollama_response
            .choices
            .first()
            .ok_or_else(|| OllamaError::Parsing("No choices in response".to_string()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .iter()
            .map(|tc| ToolCallRequest {
                function: ToolCallFunction {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                },
            })
            .collect();

        let response = ChatCompletionResponse {
            model: ollama_response.model,
            created_at: ollama_response.created.to_string(),
            message: ChatResponseMessage {
                role: match choice.message.role.as_str() {
                    "system" => ChatRole::System,
                    "user" => ChatRole::User,
                    "assistant" => ChatRole::Assistant,
                    "tool" => ChatRole::Tool,
                    _ => ChatRole::Assistant,
                },
                content: choice.message.content.clone().unwrap_or_default(),
                tool_calls,
            },
            done: choice.finish_reason.is_some(),
            done_reason: choice.finish_reason.clone().unwrap_or_default(),
            total_duration: 0,
            load_duration: 0,
            prompt_eval_count: 0,
            prompt_eval_duration: 0,
            eval_count: 0,
            eval_duration: 0,
        };

        Ok(response)
    }

    async fn chat_completion_stream(
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
        let mut body = OllamaChatRequest::from_chat_messages(messages, model)
            .set_stream(true)
            .set_ollama_options(options.into());

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OllamaTool> = self
                .tools
                .iter()
                .map(|tool| OllamaTool {
                    tool_type: "function".into(),
                    function: OpenAIStyleFunction {
                        name: tool.name().to_string(),
                        description: tool.description().to_string(),
                        parameters: tool.args_schema(),
                    },
                })
                .collect();
            body = body.set_tools(tools);
        }

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

    fn model_name(&self) -> String {
        match &self.model {
            Some(model) => model.to_string(),
            None => "".to_string(),
        }
    }

    fn tools(&self) -> &Vec<Box<dyn Tool>> {
        &self.tools
    }

    fn register_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }
}
