use std::sync::Arc;

use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::protocol::Event;
use crate::session::{Session, Task};
use autoagents_llm::LLMProvider;
use tokio::sync::mpsc;

/// Event-driven agent runner
pub struct AgentRunner<E: AgentExecutor + Send + Sync> {
    agent: BaseAgent<E>,
    tx_event: mpsc::Sender<Event>,
}

impl<E: AgentExecutor + Send + Sync> AgentRunner<E> {
    pub fn new(agent: BaseAgent<E>, tx_event: mpsc::Sender<Event>) -> Self {
        Self { agent, tx_event }
    }

    /// Run the agent in an event-driven manner
    pub async fn run(
        &self,
        llm: Arc<Box<dyn LLMProvider>>,
        mut session: Session,
        task: Task,
    ) -> Result<E::Output, E::Error> {
        // Execute the agent
        let result = self
            .agent
            .executor
            .execute(llm, &mut session, task.clone())
            .await;

        // Send completion event
        let task_result = match &result {
            Ok(value) => crate::protocol::TaskResult::Value(
                serde_json::to_value(value).unwrap_or(serde_json::Value::Null),
            ),
            Err(err) => crate::protocol::TaskResult::Failure(err.to_string()),
        };

        let _ = self
            .tx_event
            .send(Event::TaskComplete {
                sub_id: task.submission_id,
                result: task_result,
            })
            .await;

        result
    }
}
