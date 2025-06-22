use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use crate::session::{Session, SessionManager, Task};
use autoagents_llm::LLMProvider;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use uuid::Uuid;

/// Error types for Environment operations
#[derive(Debug, thiserror::Error)]
pub enum EnvironmentError {
    #[error("Agent not found: {0}")]
    AgentNotFound(Uuid),

    #[error("Task execution failed: {0}")]
    TaskExecutionFailed(String),

    #[error("Channel error: {0}")]
    ChannelError(String),

    #[error("Runtime error: {0}")]
    RuntimeError(String),

    #[error("No task set for agent: {0}")]
    NoTaskSet(Uuid),
}

/// Configuration for the Environment
#[derive(Clone)]
pub struct EnvironmentConfig {
    /// Working directory for sessions
    pub working_dir: PathBuf,
    /// Channel buffer size
    pub channel_buffer: usize,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            working_dir: std::env::current_dir().unwrap_or_default(),
            channel_buffer: 100,
        }
    }
}

/// Task handle for tracking running tasks
pub struct TaskHandle {
    pub sub_id: SubmissionId,
    pub agent_id: Uuid,
    pub join_handle: JoinHandle<Result<Value, Box<dyn std::error::Error + Send + Sync>>>,
}

/// Environment for managing agents and their execution
pub struct Environment {
    config: EnvironmentConfig,
    llm: Arc<dyn LLMProvider>,
    tx_event: mpsc::Sender<Event>,
    rx_event: Option<mpsc::Receiver<Event>>,
    sessions: HashMap<SessionId, Session>,
    default_session: SessionId,
    event_handler: Option<JoinHandle<()>>,
}

impl Environment {
    /// Create a new Environment with the given LLM and configuration
    pub fn new(llm: Arc<dyn LLMProvider>, config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let cwd = config.clone().working_dir;
        let (tx_event, rx_event) = mpsc::channel(config.channel_buffer);
        let mut session_manager = SessionManager::new(tx_event.clone());
        let session = session_manager.create_session(cwd);
        let session_id = session.id;
        let mut sessions = HashMap::new();
        sessions.insert(session_id, session);
        Self {
            config,
            llm,
            tx_event,
            rx_event: Some(rx_event),
            sessions,
            default_session: session_id,
            event_handler: None,
        }
    }

    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    pub fn get_session(&self, session_id: Option<SessionId>) -> Option<&Session> {
        if let Some(session_id) = session_id {
            self.sessions.get(&session_id)
        } else {
            self.sessions.get(&self.default_session)
        }
    }

    pub fn get_session_mut(&mut self, session_id: Option<SessionId>) -> Option<&mut Session> {
        if let Some(session_id) = session_id {
            self.sessions.get_mut(&session_id)
        } else {
            self.sessions.get_mut(&self.default_session)
        }
    }

    /// Register an agent with the environment
    pub fn register_agent<E>(
        &mut self,
        agent: BaseAgent<E>,
        session_id: Option<SessionId>,
    ) -> AgentID
    where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let agent_id = Uuid::new_v4();
        let current_session = self.get_session_mut(session_id).unwrap();
        current_session.register_agent_with_id(agent_id, agent);
        agent_id
    }

    /// Register an agent with a specific ID
    pub fn register_agent_with_id<E>(
        &mut self,
        agent_id: AgentID,
        agent: BaseAgent<E>,
        session_id: Option<SessionId>,
    ) where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let current_session = self.get_session_mut(session_id).unwrap();
        current_session.register_agent_with_id(agent_id, agent);
    }

    /// Add a task to be executed by a specific agent
    pub async fn add_task<T: Into<String>>(
        &mut self,
        agent_id: Uuid,
        task: T,
    ) -> Result<SubmissionId, EnvironmentError> {
        let task = Task::new(task, Some(agent_id));
        let sub_id = task.submission_id;

        if let Some(session) = self.sessions.get_mut(&self.default_session) {
            session.add_task(task.clone()).await;
        }

        Ok(sub_id)
    }

    /// Run a specific task
    pub async fn run_task(
        &mut self,
        agent_id: AgentID,
        sub_id: SubmissionId,
        session_id: Option<SessionId>,
    ) -> Result<Value, EnvironmentError> {
        let llm = self.llm.clone();
        let current_session = self.get_session_mut(session_id).unwrap();
        let task = current_session.get_task(sub_id);
        if task.is_none() {
            return Err(EnvironmentError::NoTaskSet(sub_id));
        }
        current_session
            .run_task(task.unwrap(), agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    /// Run the next pending task for an agent
    pub async fn run(
        &mut self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<Value, EnvironmentError> {
        let llm = self.llm.clone();
        let current_session = self.get_session_mut(session_id).unwrap();
        current_session
            .run(agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    /// Run all pending tasks for an agent
    pub async fn run_all(
        &mut self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<Vec<Value>, EnvironmentError> {
        let llm = self.llm.clone();
        let current_session = self.get_session_mut(session_id).unwrap();
        current_session
            .run_all(agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    /// Get the event sender (for advanced use cases)
    pub fn event_sender(&self) -> mpsc::Sender<Event> {
        self.tx_event.clone()
    }

    /// Get a reference to the event receiver (if not already taken)
    pub fn event_receiver(&mut self) -> Option<mpsc::Receiver<Event>> {
        self.rx_event.take()
    }

    /// Shutdown the environment gracefully
    pub async fn shutdown(&mut self) {
        //TODO Shoutdown the sessions as well

        // Stop the event handler
        if let Some(handler) = self.event_handler.take() {
            handler.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_environment_config_default() {
        let config = EnvironmentConfig::default();
        assert_eq!(config.channel_buffer, 100);
    }
}
