use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::agent::result::AgentRunResult;
use crate::agent::runnable::{RunnableAgent, RunnableAgentBuilder};
use crate::error::SessionError;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use autoagents_llm::LLMProvider;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, RwLock};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
    agent_id: Option<AgentID>,
}

impl Task {
    pub fn new<T: Into<String>>(task: T, agent_id: Option<AgentID>) -> Self {
        Self {
            prompt: task.into(),
            submission_id: Uuid::new_v4(),
            completed: false,
            result: None,
            agent_id,
        }
    }
}

#[derive(Debug, Default)]
struct State {
    current_task: Option<Task>,
    task_queue: Vec<Task>,
}

pub struct Session {
    pub id: SessionId,
    tx_event: mpsc::Sender<Event>,
    state: Arc<Mutex<State>>,
    agents: Arc<RwLock<HashMap<AgentID, Arc<dyn RunnableAgent + Send + Sync + 'static>>>>,
}

impl Session {
    pub fn new(tx_event: mpsc::Sender<Event>) -> (SessionId, Self) {
        let id = Uuid::new_v4();
        (
            id,
            Self {
                id,
                tx_event,
                state: Arc::new(Mutex::new(State::default())),
                agents: Arc::new(RwLock::new(HashMap::new())),
            },
        )
    }

    pub async fn add_task(&self, task: Task) {
        let task_clone = task.clone();
        if self
            .tx_event
            .send(Event::NewTask {
                sub_id: task_clone.submission_id,
                agent_id: task_clone.agent_id,
                prompt: task_clone.prompt,
            })
            .await
            .is_err()
        {
            panic!("Failed to send NewTask event");
        }
        let mut state = self.state.lock().await;
        state.task_queue.push(task);
    }

    pub async fn set_current_task(&self, task: Task) {
        let mut state = self.state.lock().await;
        state.current_task = Some(task);
    }

    pub async fn is_task_queue_empty(&self) -> bool {
        let state = self.state.lock().await;
        state.task_queue.is_empty()
    }

    pub async fn get_top_task(&self) -> Option<Task> {
        let mut state = self.state.lock().await;
        if state.task_queue.is_empty() {
            None
        } else {
            let task = state.task_queue.remove(0);
            state.current_task = Some(task.clone());
            Some(task)
        }
    }

    pub async fn get_current_task(&self) -> Option<Task> {
        let state = self.state.lock().await;
        state.current_task.clone()
    }

    pub async fn get_task(&self, sub_id: SubmissionId) -> Option<Task> {
        let state = self.state.lock().await;
        state
            .task_queue
            .iter()
            .find(|t| t.submission_id == sub_id)
            .cloned()
    }

    pub async fn register_agent_with_id<E>(&self, agent_id: Uuid, agent: BaseAgent<E>)
    where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let builder = RunnableAgentBuilder::new();
        let runnable_agent = builder.build(agent);
        self.agents.write().await.insert(agent_id, runnable_agent);
    }

    pub async fn run_task(
        &self,
        task: Task,
        agent_id: AgentID,
        llm: Arc<dyn LLMProvider>,
    ) -> Result<AgentRunResult, SessionError> {
        let agent = self
            .agents
            .read()
            .await
            .get(&agent_id)
            .ok_or(SessionError::AgentNotFound(agent_id))?
            .clone();
        agent
            .spawn_task(task, llm, self.tx_event.clone())
            .await
            .unwrap()
            .map_err(|e| SessionError::TaskExecutionFailed(e.to_string()))
    }

    pub async fn run(
        &self,
        agent_id: Uuid,
        llm: Arc<dyn LLMProvider>,
    ) -> Result<AgentRunResult, SessionError> {
        if let Some(task) = self.get_top_task().await {
            self.run_task(task, agent_id, llm).await
        } else {
            Err(SessionError::NoTaskSet(agent_id))
        }
    }

    pub async fn run_all(
        &self,
        agent_id: Uuid,
        llm: Arc<dyn LLMProvider>,
    ) -> Result<Vec<AgentRunResult>, SessionError> {
        let mut results = Vec::new();
        while !self.is_task_queue_empty().await {
            let result = self.run(agent_id, llm.clone()).await?;
            results.push(result);
        }
        Ok(results)
    }
}

pub struct SessionManager {
    sessions: RwLock<HashMap<SessionId, Arc<Session>>>,
    tx_event: mpsc::Sender<Event>,
}

impl SessionManager {
    pub fn new(tx_event: mpsc::Sender<Event>) -> Self {
        Self {
            sessions: RwLock::new(HashMap::new()),
            tx_event,
        }
    }

    pub async fn create_session(&self) -> (SessionId, Arc<Session>) {
        let (id, session) = Session::new(self.tx_event.clone());
        let session_arc = Arc::new(session);
        let mut sessions = self.sessions.write().await;
        sessions.insert(id, session_arc.clone());
        (id, session_arc)
    }

    pub async fn get_session(&self, session_id: &SessionId) -> Option<Arc<Session>> {
        let sessions = self.sessions.read().await;
        sessions.get(session_id).cloned()
    }
}
