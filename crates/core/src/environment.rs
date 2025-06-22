use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::agent::result::AgentRunResult;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use crate::session::{Session, SessionManager, Task};
use autoagents_llm::LLMProvider;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use uuid::Uuid;

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

#[derive(Clone)]
pub struct EnvironmentConfig {
    pub working_dir: PathBuf,
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

pub struct Environment {
    config: EnvironmentConfig,
    llm: Arc<dyn LLMProvider>,
    tx_event: mpsc::Sender<Event>,
    rx_event: Option<mpsc::Receiver<Event>>,
    session_manager: SessionManager,
    default_session: SessionId,
    event_handler: Option<JoinHandle<()>>,
}

impl Environment {
    pub async fn new(llm: Arc<dyn LLMProvider>, config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let (tx_event, rx_event) = mpsc::channel(config.channel_buffer);
        let session_manager = SessionManager::new(tx_event.clone());
        let (default_session_id, _) = session_manager.create_session().await;

        Self {
            config,
            llm,
            tx_event,
            rx_event: Some(rx_event),
            session_manager,
            default_session: default_session_id,
            event_handler: None,
        }
    }

    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    pub async fn get_session(&self, session_id: Option<SessionId>) -> Option<Arc<Session>> {
        let sid = session_id.unwrap_or(self.default_session);
        self.session_manager.get_session(&sid).await
    }

    pub async fn register_agent<E>(
        &self,
        agent: BaseAgent<E>,
        session_id: Option<SessionId>,
    ) -> AgentID
    where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let agent_id = Uuid::new_v4();
        self.register_agent_with_id(agent_id, agent, session_id)
            .await;
        agent_id
    }

    pub async fn register_agent_with_id<E>(
        &self,
        agent_id: AgentID,
        agent: BaseAgent<E>,
        session_id: Option<SessionId>,
    ) where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let session = self.get_session(session_id).await.unwrap();
        session.register_agent_with_id(agent_id, agent).await;
    }

    pub async fn add_task<T: Into<String>>(
        &self,
        agent_id: Uuid,
        task: T,
    ) -> Result<SubmissionId, EnvironmentError> {
        let task = Task::new(task, Some(agent_id));
        let sub_id = task.submission_id;
        let session = self.get_session(Some(self.default_session)).await.unwrap();
        session.add_task(task).await;
        Ok(sub_id)
    }

    pub async fn run_task(
        &self,
        agent_id: AgentID,
        sub_id: SubmissionId,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, EnvironmentError> {
        let llm = self.llm.clone();
        let session = self.get_session(session_id).await.unwrap();
        let task = session.get_task(sub_id).await;
        if task.is_none() {
            return Err(EnvironmentError::NoTaskSet(sub_id));
        }
        session
            .run_task(task.unwrap(), agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub async fn run(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, EnvironmentError> {
        let llm = self.llm.clone();
        let session = self.get_session(session_id).await.unwrap();
        session
            .run(agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub async fn run_all(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<Vec<AgentRunResult>, EnvironmentError> {
        let llm = self.llm.clone();
        let session = self.get_session(session_id).await.unwrap();
        session
            .run_all(agent_id, llm)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub fn event_sender(&self) -> mpsc::Sender<Event> {
        self.tx_event.clone()
    }

    pub fn event_receiver(&mut self) -> Option<mpsc::Receiver<Event>> {
        self.rx_event.take()
    }

    pub async fn shutdown(&mut self) {
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
