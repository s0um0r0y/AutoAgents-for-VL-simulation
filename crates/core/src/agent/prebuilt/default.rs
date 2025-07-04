//! Default Agent Implementation
//!
//! This module provides a prebuilt agent implementation that handles the common pattern of:
//! 1. Receiving a user prompt
//! 2. Processing it through an LLM (with optional tool support)
//! 3. Executing tool calls if requested by the LLM
//! 4. Continuing the conversation until a final response is generated
//!
//! # Architecture
//!
//! The default agent consists of:
//! - `Executor`: The main execution engine that manages the conversation flow
//! - `AgentBuilder`: A builder pattern for constructing agents with custom configurations
//! - `AgentOutput`: The output type containing the final response and tool calls made
//! - `AgentError`: Error types for various failure scenarios
//!
//! # Key Features
//!
//! - **Tool Support**: Automatically handles tool calls requested by the LLM
//! - **Conversation Management**: Maintains conversation history with proper message types
//! - **Turn Limiting**: Prevents infinite loops with configurable max turns
//! - **Error Handling**: Comprehensive error handling with idiomatic Rust patterns
//!
//! # Example Usage
//!
//! ```rust
//! use autoagents_core::agent::prebuilt::default::AgentBuilder;
//! use autoagents_llm::{backends::openai::OpenAI, builder::LLMBuilder};
//! use std::sync::Arc;
//!
//! let llm: Arc<OpenAI> = LLMBuilder::<OpenAI>::new()
//!         .api_key(String::from("your_api_key")) // Set the API key
//!         .model("gpt-4o") // Use GPT-4o-mini model
//!         .max_tokens(512) // Limit response length
//!         .temperature(0.2) // Control response randomness (0.0-1.0)
//!         .stream(false) // Disable streaming responses
//!         .build()
//!         .expect("Failed to build LLM");
//! let agent = AgentBuilder::new("my_agent", "You are a helpful assistant")
//!     .with_llm(llm)
//!     .build().unwrap();
//! ```
//!
//! # Tool Call Flow
//!
//! When the LLM requests a tool call:
//! 1. The tool call is recorded in the conversation history with `MessageType::ToolUse`
//! 2. The tool is executed with the provided arguments
//! 3. The result is added to the conversation with `MessageType::ToolResult`
//! 4. The conversation continues until the LLM provides a final response
//! 5. The agent loops until a tool call is present, If there is no tool call then its final response

use crate::agent::base::{AgentDeriveT, BaseAgent};
use crate::agent::executor::{AgentExecutor, TurnResult};
use crate::agent::runnable::AgentState;
use crate::session::Task;
use crate::tool::ToolCallResult;
use async_trait::async_trait;
use autoagents_llm::chat::{ChatMessage, ChatRole, MessageType, Tool};
use autoagents_llm::{LLMProvider, ToolCall, ToolT};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::Mutex;

/// A simple executor that processes user prompts and handles tool calls
///
/// The `Executor` is responsible for managing the conversation flow between the user,
/// the LLM, and any tools. It handles:
/// - Sending prompts to the LLM
/// - Processing tool call requests from the LLM
/// - Executing tools and feeding results back to the LLM
/// - Managing conversation history with proper message types
/// - Limiting the number of conversation turns to prevent infinite loops
#[derive(Clone)]
pub struct Executor {
    /// System prompt for the executor
    system_prompt: String,
    /// Maximum number of turns
    max_turns: usize,
    /// Tools available to this executor
    tools: Vec<Arc<Box<dyn ToolT>>>,
}

/// Output type for the simple executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    /// The final response from the agent
    pub response: String,
    /// Tool calls that were made
    pub tool_calls: Vec<ToolCall>,
}

/// Error type for the simple executor
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded")]
    MaxTurnsExceeded,

    #[error("JSON parsing error: {0}")]
    JsonError(#[from] serde_json::Error),
}

impl Executor {
    /// Create a new simple executor
    ///
    /// # Arguments
    /// * `system_prompt` - The system prompt that defines the agent's behavior
    /// * `tools` - Vector of tools available to the agent
    pub fn new(system_prompt: impl Into<String>, tools: Vec<Arc<Box<dyn ToolT>>>) -> Self {
        Self {
            system_prompt: system_prompt.into(),
            max_turns: 10,
            tools,
        }
    }

    /// Set the maximum number of turns
    ///
    /// This limits how many back-and-forth interactions can occur between
    /// the LLM and tools before the conversation must complete.
    ///
    /// # Arguments
    /// * `max_turns` - Maximum number of conversation turns allowed
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Process tool calls from the LLM response
    ///
    /// Takes a vector of tool calls from the LLM and executes the first matching tool.
    /// Returns the result of the tool execution wrapped in a `ToolCallResult`.
    ///
    /// # Arguments
    /// * `tool_calls` - Vector of tool calls requested by the LLM
    ///
    /// # Returns
    /// * `Ok(Some(ToolCallResult))` - If a tool was found and executed successfully
    /// * `Ok(None)` - If no matching tool was found
    /// * `Err(AgentError::JsonError)` - If tool arguments couldn't be parsed
    async fn process_tool_calls(
        &self,
        tool_calls: Vec<ToolCall>,
    ) -> Result<Option<ToolCallResult>, AgentError> {
        // // Process each tool call
        if let Some(tool) = tool_calls.first() {
            let arguments = tool.function.arguments.clone();
            for tool_self in self.tools.clone() {
                if tool.function.name == tool_self.name() {
                    let result = tool_self.run(serde_json::from_str(&arguments.clone())?);
                    // let result = llm.call_tool(tool_name, arguments.clone());
                    return Ok(Some(ToolCallResult {
                        tool_name: tool.function.name.clone(),
                        arguments: arguments.clone().into(),
                        result,
                    }));
                }
            }
        }

        Ok(None)
    }
}

#[async_trait]
impl AgentExecutor for Executor {
    type Output = Value;
    type Error = AgentError;

    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        task: Task,
        state: Arc<Mutex<AgentState>>,
    ) -> Result<Self::Output, Self::Error> {
        // Initialize conversation with system prompt and task
        let mut previous_chat_history = state.lock().await.get_history().messages;
        let messages = vec![
            ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::Text,
                content: self.system_prompt.clone(),
            },
            ChatMessage {
                role: ChatRole::User,
                message_type: MessageType::Text,
                content: task.prompt,
            },
        ];
        previous_chat_history.extend(messages);

        let mut turn_count = 0;

        // Process turns until we get a final response or hit the limit
        loop {
            turn_count += 1;
            if turn_count > self.max_turns {
                return Err(AgentError::MaxTurnsExceeded);
            }

            match self
                .process_turn(llm.clone(), &mut previous_chat_history, state.clone())
                .await?
            {
                TurnResult::Complete(output) => return Ok(output),
                TurnResult::Continue => continue,
                TurnResult::Error(e) => {
                    // Add error to conversation and continue
                    previous_chat_history.push(ChatMessage {
                        role: ChatRole::Assistant,
                        message_type: MessageType::Text,
                        content: format!("Error occurred: {}", e),
                    });
                }
            }
        }
    }

    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        messages: &mut Vec<ChatMessage>,
        state: Arc<Mutex<AgentState>>,
    ) -> Result<TurnResult<Self::Output>, Self::Error> {
        let has_tools = !self.tools.is_empty();
        let response = if has_tools {
            let tools = self.tools.iter().map(Tool::from).collect::<Vec<_>>();
            llm.chat_with_tools(messages.as_slice(), Some(tools.as_slice()))
                .await
        } else {
            // Call the LLM
            llm.chat(messages.as_slice()).await
        };

        let response_mapped = response.map_err(|e| AgentError::LLMError(e.to_string()))?;

        // Check if the response contains tool calls
        let final_response = if response_mapped.tool_calls().is_none() {
            // No tool calls, this is the final response
            response_mapped
                .text()
                .clone()
                .ok_or_else(|| AgentError::LLMError("No text in response".to_string()))?
        } else {
            let tool_calls = response_mapped.tool_calls().ok_or_else(|| {
                AgentError::LLMError("Tool calls expected but none found".to_string())
            })?;

            // Add the assistant's message with tool calls to the conversation
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                message_type: MessageType::ToolUse(tool_calls.clone()),
                content: response_mapped.text().clone().unwrap_or_default(),
            });

            // Process tool calls and get results
            let result = self.process_tool_calls(tool_calls.clone()).await?;

            if let Some(tool_result) = result {
                // Add tool result as a new message with proper MessageType
                messages.push(ChatMessage {
                    role: ChatRole::Assistant,
                    message_type: MessageType::ToolResult(tool_calls),
                    content: tool_result.result.to_string(),
                });

                state.lock().await.record_tool_call(tool_result);
            } else {
                return Err(AgentError::ToolError(
                    "Tool call processed but no result returned".to_string(),
                ));
            }
            // Continue the conversation after tool calls
            return Ok(TurnResult::Continue);
        };
        // Record the final message
        state.lock().await.record_conversation(ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: final_response.clone(),
        });

        Ok(TurnResult::Complete(final_response.into()))
    }

    fn max_turns(&self) -> usize {
        self.max_turns
    }
}

/// Builder for creating default agents with a fluent API
///
/// The `AgentBuilder` provides a convenient way to construct agents with:
/// - Custom system prompts
/// - Tool configurations
/// - LLM providers
/// - Other agent settings
pub struct AgentBuilder {
    name: String,
    description: String,
    tools: Vec<Arc<Box<dyn ToolT>>>,
    llm: Option<Arc<dyn LLMProvider>>,
}

impl AgentBuilder {
    /// Create a new agent builder with a name and system prompt
    ///
    /// # Arguments
    /// * `name` - The name of the agent
    /// * `system_prompt` - The system prompt that defines the agent's behavior
    pub fn new(name: impl Into<String>, system_prompt: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: system_prompt.into(),
            tools: Vec::new(),
            llm: None,
        }
    }

    /// Create a builder from an existing agent implementation
    ///
    /// This method extracts the configuration from an agent that implements
    /// `AgentDeriveT` and creates a builder with those settings.
    ///
    /// # Arguments
    /// * `agent` - An agent implementing the `AgentDeriveT` trait
    pub fn from_agent<T: AgentDeriveT>(agent: T) -> Self {
        let tools: Vec<Arc<Box<dyn ToolT>>> = agent
            .tools()
            .into_iter()
            .map(|tool| {
                let boxed: Box<dyn ToolT> = tool;
                let arc: Arc<Box<dyn ToolT>> = Arc::new(boxed);
                arc
            })
            .collect();
        Self {
            name: agent.name().into(),
            description: agent.description().into(),
            tools,
            llm: None,
        }
    }

    /// Set the LLM provider for the agent
    ///
    /// # Arguments
    /// * `llm` - The LLM provider to use for generating responses
    pub fn with_llm(mut self, llm: Arc<dyn LLMProvider>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// Build the agent with the configured settings
    ///
    /// # Returns
    /// * `Ok(BaseAgent<Executor>)` - The constructed agent
    /// * `Err(AgentError::LLMError)` - If no LLM provider was set
    pub fn build(self) -> Result<BaseAgent<Executor>, AgentError> {
        let llm = self
            .llm
            .ok_or_else(|| AgentError::LLMError("LLm is not set".into()))?;
        Ok(BaseAgent::new(
            self.name,
            self.description.clone(),
            Executor::new(self.description, self.tools.clone()),
            self.tools,
            llm,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_executor_creation() {
        let executor =
            Executor::new("You are a helpful assistant".to_string(), vec![]).with_max_turns(5);

        assert_eq!(executor.max_turns(), 5);
    }
}
