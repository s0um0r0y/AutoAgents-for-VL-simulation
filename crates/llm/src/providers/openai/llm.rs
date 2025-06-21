use super::{api::OpenAIAPI, model::OpenAIModel};
use crate::{
    common::openai_types::{
        parse_openai_style_stream_chunk, OpenAIStyleChatCompletionOptions,
        OpenAIStyleChatCompletionRequest, OpenAIStyleChatCompletionResponse, OpenAIStyleFunction,
        OpenAIStyleStreamResponse, OpenAIStyleTextGenerationOptions, OpenAIStyleTool,
    },
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
use serde_json::Value;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OpenAIError {
    #[error("API error: {0}")]
    Api(String),
    #[error("Parsing error: {0}")]
    Parsing(String),
    #[error("Authentication error: API key not set")]
    Authentication,
    #[error("LLM Error: {0}")]
    GenericLLMError(#[from] LLMError),
}

impl LLMProviderError for OpenAIError {}

#[derive(Debug)]
pub struct OpenAI {
    base_url: String,
    api_key: Option<String>,
    pub model: Option<OpenAIModel>,
    pub tools: Vec<Box<dyn Tool>>,
}

impl Default for OpenAI {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com".into(),
            api_key: std::env::var("OPENAI_API_KEY").ok(),
            model: Some(OpenAIModel::GPT35Turbo),
            tools: vec![],
        }
    }
}

impl Clone for OpenAI {
    fn clone(&self) -> Self {
        Self {
            base_url: self.base_url.clone(),
            api_key: self.api_key.clone(),
            model: self.model.clone(),
            tools: vec![], // Tools cannot be cloned, so we start with an empty vec
        }
    }
}

impl OpenAI {
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

    fn api_key(&self) -> Result<String, OpenAIError> {
        self.api_key.clone().ok_or(OpenAIError::Authentication)
    }

    pub fn set_base_url<T: Into<String>>(mut self, base_url: T) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn set_api_key<T: Into<String>>(mut self, api_key: T) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn set_model(mut self, model: OpenAIModel) -> Self {
        self.model = Some(model);
        self
    }

    async fn make_request(&self, url: &str, body: Value) -> Result<String, OpenAIError> {
        let api_key = self.api_key()?;
        let headers = vec![("Authorization".to_string(), format!("Bearer {}", api_key))];

        HTTPRequest::request_with_headers(url, body, headers)
            .await
            .map_err(|e| OpenAIError::Api(e.to_string()))
    }

    fn make_request_sync(&self, url: &str, body: Value) -> Result<String, OpenAIError> {
        let api_key = self.api_key()?;
        let headers = vec![("Authorization".to_string(), format!("Bearer {}", api_key))];

        HTTPRequest::request_sync_with_headers(url, body, headers)
            .map_err(|e| OpenAIError::Api(e.to_string()))
    }
}

#[async_trait]
impl LLM for OpenAI {
    type Error = OpenAIError;

    fn name(&self) -> &'static str {
        "OpenAI"
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
        let options: OpenAIStyleTextGenerationOptions = options.unwrap_or_default().into();

        // For text generation, we'll use the chat completions API with system and user messages
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt.to_string(),
            },
        ];

        let chat_options = ChatCompletionOptions {
            num_tokens: options.max_tokens,
        };

        let response = self.chat_completion(messages, Some(chat_options)).await?;

        Ok(TextGenerationResponse {
            response: response.message.content,
        })
    }

    async fn text_generation_stream(
        &self,
        prompt: &str,
        system_prompt: &str,
        options: Option<TextGenerationOptions>,
    ) -> BoxStream<'static, Result<TextGenerationResponse, OpenAIError>> {
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: prompt.to_string(),
            },
        ];

        let chat_options = options.map(|opts| ChatCompletionOptions {
            num_tokens: opts.num_tokens,
        });

        let stream = self.chat_completion_stream(messages, chat_options).await;
        stream
            .map(|result| {
                result.map(|response| TextGenerationResponse {
                    response: response.message.content,
                })
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
        let options: OpenAIStyleChatCompletionOptions = options.unwrap_or_default().into();

        let mut body = OpenAIStyleChatCompletionRequest::from_chat_messages(messages, model)
            .set_chat_options(options);

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OpenAIStyleTool> = self
                .tools
                .iter()
                .map(|tool| OpenAIStyleTool {
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

        let url = utils::create_model_url(base_url, OpenAIAPI::ChatCompletion);
        let text = self.make_request(&url, serde_json::json!(body)).await?;

        let openai_response: OpenAIStyleChatCompletionResponse =
            serde_json::from_str(&text).map_err(|e| OpenAIError::Parsing(e.to_string()))?;

        // Convert OpenAI response to our common format
        let choice = openai_response
            .choices
            .first()
            .ok_or_else(|| OpenAIError::Parsing("No choices in response".to_string()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .iter()
            .map(|tc| ToolCallRequest {
                function: ToolCallFunction {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::Object(serde_json::Map::new())),
                },
            })
            .collect();

        let response = ChatCompletionResponse {
            model: openai_response.model,
            created_at: openai_response.created.to_string(),
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
        let options: OpenAIStyleChatCompletionOptions = options.unwrap_or_default().into();

        let mut body = OpenAIStyleChatCompletionRequest::from_chat_messages(messages, model)
            .set_chat_options(options);

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OpenAIStyleTool> = self
                .tools
                .iter()
                .map(|tool| OpenAIStyleTool {
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

        let url = utils::create_model_url(base_url, OpenAIAPI::ChatCompletion);
        let text = self.make_request_sync(&url, serde_json::json!(body))?;

        let openai_response: OpenAIStyleChatCompletionResponse =
            serde_json::from_str(&text).map_err(|e| OpenAIError::Parsing(e.to_string()))?;

        // Convert OpenAI response to our common format
        let choice = openai_response
            .choices
            .first()
            .ok_or_else(|| OpenAIError::Parsing("No choices in response".to_string()))?;

        let tool_calls = choice
            .message
            .tool_calls
            .iter()
            .map(|tc| ToolCallRequest {
                function: ToolCallFunction {
                    name: tc.function.name.clone(),
                    arguments: serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::Object(serde_json::Map::new())),
                },
            })
            .collect();

        let response = ChatCompletionResponse {
            model: openai_response.model,
            created_at: openai_response.created.to_string(),
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
    ) -> BoxStream<'static, Result<ChatCompletionResponse, OpenAIError>> {
        let model = match self.model() {
            Ok(m) => m,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };
        let base_url = match self.base_url() {
            Ok(url) => url,
            Err(e) => return stream::once(async { Err(e.into()) }).boxed(),
        };
        let api_key = match self.api_key() {
            Ok(key) => key,
            Err(e) => return stream::once(async { Err(e) }).boxed(),
        };

        let options: OpenAIStyleChatCompletionOptions = options.unwrap_or_default().into();
        let mut body = OpenAIStyleChatCompletionRequest::from_chat_messages(messages, model)
            .set_stream(true)
            .set_chat_options(options);

        // Add any registered tools from the LLM into the chat request.
        if !self.tools.is_empty() {
            let tools: Vec<OpenAIStyleTool> = self
                .tools
                .iter()
                .map(|tool| OpenAIStyleTool {
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

        let url = utils::create_model_url(base_url, OpenAIAPI::ChatCompletion);
        let headers = vec![("Authorization".to_string(), format!("Bearer {}", api_key))];

        let stream_future =
            HTTPRequest::stream_request_with_headers(url, serde_json::json!(body), headers);

        stream::once(stream_future)
            .then(|result| async move {
                match result {
                    Ok(inner_stream) => inner_stream
                        .map_err(|e| OpenAIError::Api(e.to_string()))
                        .boxed(),
                    Err(e) => stream::iter(vec![Err(OpenAIError::Api(e.to_string()))]).boxed(),
                }
            })
            .flatten()
            .filter_map(|chunk_result| async move {
                match chunk_result {
                    Ok(chunk) => {
                        // Parse OpenAI streaming format
                        if let Some(data) = parse_openai_style_stream_chunk(&chunk) {
                            match serde_json::from_str::<OpenAIStyleStreamResponse>(&data) {
                                Ok(stream_response) => {
                                    if let Some(choice) = stream_response.choices.first() {
                                        let response = ChatCompletionResponse {
                                            model: stream_response.model,
                                            created_at: stream_response.created.to_string(),
                                            message: ChatResponseMessage {
                                                role: match choice
                                                    .delta
                                                    .role
                                                    .as_deref()
                                                    .unwrap_or("assistant")
                                                {
                                                    "system" => ChatRole::System,
                                                    "user" => ChatRole::User,
                                                    "assistant" => ChatRole::Assistant,
                                                    "tool" => ChatRole::Tool,
                                                    _ => ChatRole::Assistant,
                                                },
                                                content: choice
                                                    .delta
                                                    .content
                                                    .clone()
                                                    .unwrap_or_default(),
                                                tool_calls: vec![], // TODO: Handle tool calls in streaming
                                            },
                                            done: choice.finish_reason.is_some(),
                                            done_reason: choice
                                                .finish_reason
                                                .clone()
                                                .unwrap_or_default(),
                                            total_duration: 0,
                                            load_duration: 0,
                                            prompt_eval_count: 0,
                                            prompt_eval_duration: 0,
                                            eval_count: 0,
                                            eval_duration: 0,
                                        };
                                        Some(Ok(response))
                                    } else {
                                        None
                                    }
                                }
                                Err(e) => Some(Err(OpenAIError::Parsing(e.to_string()))),
                            }
                        } else {
                            None
                        }
                    }
                    Err(e) => Some(Err(e)),
                }
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
