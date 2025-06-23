use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::agent::result::AgentRunResult;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use crate::session::{Session, SessionManager, Task};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;
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
    session_manager: SessionManager,
    default_session: SessionId,
    event_handler: Option<JoinHandle<()>>,
}

impl Environment {
    pub async fn new(config: Option<EnvironmentConfig>) -> Self {
        let config = config.unwrap_or_default();
        let session_manager = SessionManager::new();
        let (default_session_id, _) = session_manager.create_session(config.channel_buffer).await;

        Self {
            config,
            session_manager,
            default_session: default_session_id,
            event_handler: None,
        }
    }

    pub fn config(&self) -> &EnvironmentConfig {
        &self.config
    }

    pub async fn get_session(&self, session_id: Option<SessionId>) -> Option<Arc<Mutex<Session>>> {
        let sid = session_id.unwrap_or(self.default_session);
        self.session_manager.get_session(&sid).await
    }

    pub async fn get_session_mut(
        &self,
        session_id: Option<SessionId>,
    ) -> Option<Arc<Mutex<Session>>> {
        let sid = session_id.unwrap_or(self.default_session);
        self.session_manager.get_session_mut(&sid).await
    }

    pub async fn register_agent<E>(
        &self,
        agent: BaseAgent<E>,
        session_id: Option<SessionId>,
    ) -> AgentID
    where
        E: AgentExecutor,
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
        E: AgentExecutor,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let session_arc = self
            .get_session(session_id)
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        session.register_agent_with_id(agent_id, agent).await;
    }

    pub async fn add_task<T: Into<String>>(
        &self,
        agent_id: Uuid,
        task: T,
    ) -> Result<SubmissionId, EnvironmentError> {
        let task = Task::new(task, Some(agent_id));
        let sub_id = task.submission_id;
        let session_arc = self
            .get_session(Some(self.default_session))
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        session.add_task(task).await;
        Ok(sub_id)
    }

    pub async fn run_task(
        &self,
        agent_id: AgentID,
        sub_id: SubmissionId,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, EnvironmentError> {
        let session_arc = self
            .get_session(session_id)
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        let task = session.get_task(sub_id).await;
        if task.is_none() {
            return Err(EnvironmentError::NoTaskSet(sub_id));
        }
        session
            .run_task(task.unwrap(), agent_id)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub async fn run(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<AgentRunResult, EnvironmentError> {
        let session_arc = self
            .get_session(session_id)
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        session
            .run(agent_id)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub async fn run_all(
        &self,
        agent_id: AgentID,
        session_id: Option<SessionId>,
    ) -> Result<Vec<AgentRunResult>, EnvironmentError> {
        let session_arc = self
            .get_session(session_id)
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        session
            .run_all(agent_id)
            .await
            .map_err(|e| EnvironmentError::RuntimeError(e.to_string()))
    }

    pub async fn event_sender(&self, session_id: Option<SessionId>) -> mpsc::Sender<Event> {
        let session_arc = self
            .get_session(session_id)
            .await
            .expect("Session not found");
        let session = session_arc.lock().await;
        session.event_sender().clone()
    }

    pub async fn take_event_receiver(
        &mut self,
        session_id: Option<SessionId>,
    ) -> Option<ReceiverStream<Event>> {
        if let Some(session_arc_mutex) = self.get_session_mut(session_id).await {
            let mut session = session_arc_mutex.lock().await;
            session.take_event_receiver()
        } else {
            None
        }
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
