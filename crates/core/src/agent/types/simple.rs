use crate::agent::base::{AgentDeriveT, BaseAgent};
use crate::agent::executor::{AgentExecutor, TurnResult};
use crate::session::{Session, Task, ToolCall};
use async_trait::async_trait;
use autoagents_llm::llm::{ChatCompletionResponse, ChatMessage, ChatRole, LLM};
use autoagents_llm::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

/// A simple executor that processes user prompts and handles tool calls
#[derive(Clone)]
pub struct SimpleExecutor {
    /// System prompt for the executor
    system_prompt: String,
    /// Maximum number of turns
    max_turns: usize,
    /// Tools available to this executor
    tools: Vec<Arc<dyn Tool>>,
}

/// Output type for the simple executor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimpleOutput {
    /// The final response from the agent
    pub response: String,
    /// Tool calls that were made
    pub tool_calls: Vec<ToolCall>,
}

/// Error type for the simple executor
#[derive(Debug, thiserror::Error)]
pub enum SimpleError {
    #[error("LLM error: {0}")]
    LLMError(String),

    #[error("Session error: {0}")]
    SessionError(String),

    #[error("Tool execution error: {0}")]
    ToolError(String),

    #[error("Maximum turns exceeded")]
    MaxTurnsExceeded,
}

impl SimpleExecutor {
    /// Create a new simple executor
    pub fn new(system_prompt: String) -> Self {
        Self {
            system_prompt,
            max_turns: 10,
            tools: vec![],
        }
    }

    /// Set the maximum number of turns
    pub fn with_max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = max_turns;
        self
    }

    /// Add a tool to this executor
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Process tool calls from the LLM response
    async fn process_tool_calls<T: LLM>(
        &self,
        llm: &T,
        response: &ChatCompletionResponse,
        session: &mut Session,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<Option<ToolCall>, SimpleError> {
        // Add the assistant's message to the conversation
        let assistant_message = ChatMessage {
            role: ChatRole::Assistant,
            content: response.message.content.clone(),
        };

        session.record_conversation(assistant_message.clone());
        messages.push(assistant_message);

        // Process each tool call
        if let Some(tool_call) = &response.message.tool_calls.first() {
            let tool_name = &tool_call.function.name;
            let arguments = &tool_call.function.arguments;
            let result = llm.call_tool(tool_name, arguments.clone());
            return Ok(Some(ToolCall {
                tool_name: tool_name.clone(),
                arguments: arguments.clone(),
                result: result.unwrap(),
            }));
        }

        Ok(None)
    }
}

#[async_trait]
impl AgentExecutor for SimpleExecutor {
    type Output = Value;
    type Error = SimpleError;

    async fn execute<L: LLM + Clone + 'static>(
        &self,
        llm: &L,
        session: &mut Session,
        task: Task,
    ) -> Result<Self::Output, Self::Error> {
        // Initialize conversation with system prompt and task
        let mut previous_chat_history = session.get_history().messages.clone();
        let messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: self.system_prompt.clone(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: task.prompt,
            },
        ];
        previous_chat_history.extend(messages);

        let mut turn_count = 0;

        // Process turns until we get a final response or hit the limit
        loop {
            turn_count += 1;
            if turn_count > self.max_turns {
                return Err(SimpleError::MaxTurnsExceeded);
            }

            match self
                .process_turn(llm, session, &mut previous_chat_history)
                .await?
            {
                TurnResult::Complete(output) => return Ok(output),
                TurnResult::Continue => continue,
                TurnResult::Error(e) => {
                    // Add error to conversation and continue
                    previous_chat_history.push(ChatMessage {
                        role: ChatRole::System,
                        content: format!("Error occurred: {}", e),
                    });
                }
            }
        }
    }

    async fn process_turn<L: LLM + Clone + 'static>(
        &self,
        llm: &L,
        session: &mut Session,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<TurnResult<Self::Output>, Self::Error> {
        // Call the LLM
        let response = llm
            .chat_completion(messages.clone(), None)
            .await
            .map_err(|e| SimpleError::LLMError(e.to_string()))?;

        // Check if the response contains tool calls
        if !response.message.tool_calls.is_empty() {
            let mut tool_calls = Vec::new();
            let result = self
                .process_tool_calls(llm, &response, session, messages)
                .await?;
            let restult_string = result.clone().unwrap().result.to_string();
            tool_calls.push(result);
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: restult_string,
            });
            // Continue the conversation after tool calls
            Ok(TurnResult::Continue)
        } else {
            // No tool calls, this is the final response
            let final_response = response.message.content.clone();

            // Record the final message
            session.record_conversation(ChatMessage {
                role: ChatRole::Assistant,
                content: final_response.clone(),
            });

            Ok(TurnResult::Complete(final_response.into()))
        }
    }

    fn max_turns(&self) -> usize {
        self.max_turns
    }
}

/// Builder for creating Simple agents
pub struct SimpleAgentBuilder {
    name: String,
    description: String,
    executor: SimpleExecutor,
    tools: Vec<Arc<dyn Tool>>,
}

impl SimpleAgentBuilder {
    pub fn new(name: impl Into<String>, system_prompt: String) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            executor: SimpleExecutor::new(system_prompt),
            tools: Vec::new(),
        }
    }

    pub fn from_agent<T: AgentDeriveT>(a: T, system_prompt: String) -> Self {
        Self {
            name: a.name().into(),
            description: a.description().into(),
            executor: SimpleExecutor::new(system_prompt),
            tools: a
                .tools()
                .into_iter()
                .map(|tool| {
                    let boxed: Box<dyn Tool> = tool;
                    let arc: Arc<dyn Tool> = boxed.into();
                    arc
                })
                .collect(),
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn build(self) -> BaseAgent<SimpleExecutor> {
        BaseAgent::new(self.name, self.description, self.executor, self.tools)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_executor_creation() {
        let executor =
            SimpleExecutor::new("You are a helpful assistant".to_string()).with_max_turns(5);

        assert_eq!(executor.max_turns(), 5);
    }
}
