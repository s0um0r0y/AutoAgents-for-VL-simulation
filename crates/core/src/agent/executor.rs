use crate::session::{Session, Task};
use async_trait::async_trait;
use autoagents_llm::chat::ChatMessage;
use autoagents_llm::LLMProvider;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

/// Result of processing a single turn
#[derive(Debug)]
pub enum TurnResult<T> {
    /// Continue processing
    Continue,
    /// Final result obtained
    Complete(T),
    /// Error occurred but can continue
    Error(String),
}

/// Base trait for agent execution strategies
#[async_trait]
pub trait AgentExecutor: Send + Sync + 'static {
    type Output: Serialize + DeserializeOwned + Send + Sync;
    type Error: Error + Send + Sync + 'static;

    /// Execute the agent with the given session
    async fn execute(
        &self,
        llm: Arc<dyn LLMProvider>,
        session: &mut Session,
        task: Task,
    ) -> Result<Self::Output, Self::Error>;

    /// Process a single turn of conversation
    async fn process_turn(
        &self,
        llm: Arc<dyn LLMProvider>,
        session: &mut Session,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<TurnResult<Self::Output>, Self::Error>;

    /// Get the maximum number of turns allowed
    fn max_turns(&self) -> usize {
        10
    }
}
