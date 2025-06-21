use crate::agent::base::BaseAgent;
use crate::agent::executor::AgentExecutor;
use crate::protocol::Event;
use crate::session::{Session, Task};
use autoagents_llm::llm::LLM;
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
    pub async fn run<L: LLM + Clone + 'static>(
        &self,
        mut llm: L,
        mut session: Session,
        task: Task,
    ) -> Result<E::Output, E::Error> {
        // Register tools with the LLM
        if let Err(e) = self.agent.register_tools(&mut llm) {
            let error_msg = e.to_string();
            let task_result = crate::protocol::TaskResult::Failure(error_msg);
            let _ = self
                .tx_event
                .send(Event::TaskComplete {
                    sub_id: task.submission_id,
                    result: task_result,
                })
                .await;

            // For now, we'll panic as we don't have a generic way to convert errors
            // In a real implementation, AgentExecutor::Error should have a way to handle AgentError
            panic!("Failed to register tools: {}", e);
        }

        // Execute the agent
        let result = self
            .agent
            .executor
            .execute(&llm, &mut session, task.clone())
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
