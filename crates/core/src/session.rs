use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::agent::runnable::{RunnableAgent, RunnableAgentBuilder};
use crate::error::SessionError;
use crate::protocol::{AgentID, Event, SessionId, SubmissionId};
use autoagents_llm::chat::ChatMessage;
use autoagents_llm::LLMProvider;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub prompt: String,
    pub submission_id: SubmissionId,
    pub completed: bool,
    pub result: Option<Value>,
    /// Task executed by the Agent ID
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

/// Record of a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: Value,
    pub result: Value,
}

/// Session history tracks agent interactions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct History {
    /// Chat messages in the conversation
    pub messages: Vec<ChatMessage>,
    /// Tool calls that have been made
    pub tool_calls: Vec<ToolCall>,
    /// Tasks that were created
    pub tasks: Vec<Task>,
}

/// Session state
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
struct State {
    /// The currently running task
    current_task: Option<Task>,
    ///Tasks Queue
    task_queue: Vec<Task>,
    /// Session history
    history: History,
}

#[derive(Clone)]
/// A session represents an interaction with an agent
pub struct Session {
    ///Session ID
    pub id: SessionId,
    /// Event sender for communicating with the runtime
    tx_event: mpsc::Sender<Event>,
    /// Current working directory
    cwd: PathBuf,
    /// State for the session
    state: Arc<Mutex<State>>,
    agents: HashMap<AgentID, Arc<dyn RunnableAgent>>,
}

impl Session {
    /// Create a new session
    pub fn new(tx_event: mpsc::Sender<Event>, cwd: PathBuf) -> Self {
        Self {
            id: Uuid::new_v4(),
            tx_event,
            cwd,
            state: Arc::new(Mutex::new(State::default())),
            agents: HashMap::new(),
        }
    }

    /// Resolve a path relative to the current working directory
    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            path
        } else {
            self.cwd.join(path)
        }
    }

    /// Set the task
    pub async fn add_task(&mut self, task: Task) {
        let mut state = self.state.lock().unwrap();
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
        state.task_queue.push(task);
    }

    pub fn set_current_task(&mut self, task: Task) {
        let mut state = self.state.lock().unwrap();
        state.current_task = Some(task);
    }

    pub fn is_task_queue_empty(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.task_queue.is_empty()
    }

    ///Get the task
    pub fn get_top_task(&mut self) -> Option<Task> {
        let mut state = self.state.lock().unwrap();
        if state.task_queue.is_empty() {
            None
        } else {
            let task = state.task_queue.remove(0);
            state.current_task = Some(task.clone());
            Some(task)
        }
    }

    /// Send an event to the runtime
    pub async fn send_event(&self, event: Event) -> Result<(), String> {
        self.tx_event
            .send(event)
            .await
            .map_err(|e| format!("Failed to send event: {}", e))
    }

    /// Record a conversation item in history
    pub fn record_conversation(&mut self, message: ChatMessage) {
        let mut state = self.state.lock().unwrap();
        state.history.messages.push(message);
    }

    /// Record a tool call in history
    pub fn record_tool_call(&mut self, tool_call: ToolCall) {
        let mut state = self.state.lock().unwrap();
        state.history.tool_calls.push(tool_call);
    }

    /// Get the current history
    pub fn get_history(&self) -> History {
        let state = self.state.lock().unwrap();
        state.history.clone()
    }

    /// Get the current task
    pub fn get_current_task(&self) -> Option<Task> {
        let state = self.state.lock().unwrap();
        state.current_task.clone()
    }

    /// Get the current task
    pub fn get_task(&self, sub_id: SubmissionId) -> Option<Task> {
        let state = self.state.lock().unwrap();
        state
            .task_queue
            .iter()
            .find(|t| t.submission_id == sub_id)
            .cloned()
    }

    /// Register an agent with a specific ID
    pub fn register_agent_with_id<E>(&mut self, agent_id: Uuid, agent: BaseAgent<E>)
    where
        E: AgentExecutor + Clone + Send + Sync + 'static,
        E::Output: Into<Value> + Send,
        E::Error: std::error::Error + Send + Sync + 'static,
    {
        let builder = RunnableAgentBuilder::new();
        let runnable_agent = builder.build(agent);
        self.agents.insert(agent_id, runnable_agent);
    }

    /// Run a specific task
    pub async fn run_task(
        &mut self,
        task: Task,
        agent_id: AgentID,
        llm: Arc<Box<dyn LLMProvider>>,
    ) -> Result<Value, SessionError> {
        // Get the agent
        let agent = self
            .agents
            .get(&agent_id)
            .ok_or(SessionError::AgentNotFound(agent_id))?
            .clone();

        // Run the agent
        // TODO - Add JoinHandle to the queue to  handle them, Currently one task is processed at a time
        agent
            .spawn_task(self.clone(), task, llm, self.tx_event.clone())
            .await
            .unwrap()
            .map_err(|e| SessionError::TaskExecutionFailed(e.to_string()))
    }

    /// Run the next pending task for an agent
    pub async fn run(
        &mut self,
        agent_id: Uuid,
        llm: Arc<Box<dyn LLMProvider>>,
    ) -> Result<Value, SessionError> {
        if let Some(task) = self.get_top_task() {
            self.run_task(task, agent_id, llm).await
        } else {
            Err(SessionError::NoTaskSet(agent_id))
        }
    }

    /// Run all pending tasks for an agent
    pub async fn run_all(
        &mut self,
        agent_id: Uuid,
        llm: Arc<Box<dyn LLMProvider>>,
    ) -> Result<Vec<Value>, SessionError> {
        let mut results = Vec::new();

        // Keep running until no more tasks
        while !self.is_task_queue_empty() {
            let result = self.run(agent_id, llm.clone()).await?;
            results.push(result);
        }

        Ok(results)
    }

    //TODO
    pub async fn shutdown(&mut self) {
        todo!()
    }

    /// Wait for all tasks to complete
    //TODO
    pub async fn wait_all(&mut self) -> HashMap<SubmissionId, Result<Value, String>> {
        todo!()
    }
}

/// The session manager handles creating and managing sessions
pub struct SessionManager {
    sessions: HashMap<Uuid, Session>,
    tx_event: mpsc::Sender<Event>,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(tx_event: mpsc::Sender<Event>) -> Self {
        Self {
            sessions: HashMap::new(),
            tx_event,
        }
    }

    /// Create a new session for an agent
    pub fn create_session(&mut self, cwd: PathBuf) -> Session {
        let session = Session::new(self.tx_event.clone(), cwd);
        self.sessions.insert(session.id, session.clone());
        session
    }

    /// Get a session for an agent
    pub fn get_session(&self, session_id: &Uuid) -> Option<&Session> {
        self.sessions.get(session_id)
    }

    /// Get a mutable session for an agent
    pub fn get_session_mut(&mut self, session_id: &Uuid) -> Option<&mut Session> {
        self.sessions.get_mut(session_id)
    }

    /// Remove a session for an agent
    pub fn remove_session(&mut self, session_id: &Uuid) -> Option<Session> {
        self.sessions.remove(session_id)
    }
}
