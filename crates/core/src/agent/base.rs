use super::executor::AgentExecutor;
use async_trait::async_trait;
use autoagents_llm::ToolT;
use std::sync::Arc;

/// Base agent type that uses an executor strategy
#[derive(Clone)]
pub struct BaseAgent<E: AgentExecutor + Send + Sync> {
    pub name: String,
    pub description: String,
    pub executor: E,
    pub tools: Vec<Arc<Box<dyn ToolT>>>,
}

impl<E: AgentExecutor> BaseAgent<E> {
    pub fn new(
        name: String,
        description: String,
        executor: E,
        tools: Vec<Arc<Box<dyn ToolT>>>,
    ) -> Self {
        Self {
            name,
            description,
            executor,
            tools,
        }
    }

    /// Add a tool to this agent
    pub fn with_tool(mut self, tool: Arc<Box<dyn ToolT>>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to this agent
    pub fn with_tools(mut self, tools: Vec<Arc<Box<dyn ToolT>>>) -> Self {
        self.tools.extend(tools);
        self
    }
}

#[async_trait]
pub trait AgentDeriveT: Send + Sync {
    fn description(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn tools(&self) -> Vec<Box<dyn ToolT>>;
}

// #[cfg(test)]
// mod tests {
//     #![allow(dead_code)]
//     use super::*;
//     use crate::session::Session;
//     use crate::{agent::executor::TurnResult, session::Task};
//     use async_trait::async_trait;
//     use autoagents_llm::chat::ChatMessage;
//     use serde::{Deserialize, Serialize};
//     use serde_json::Value;
//     use std::sync::Arc;

//     // Mock tool for testing
//     #[derive(Debug, Clone)]
//     struct MockTool {
//         name: String,
//         description: String,
//     }

//     impl ToolT for MockTool {
//         fn name(&self) -> &'static str {
//             Box::leak(self.name.clone().into_boxed_str())
//         }

//         fn description(&self) -> &'static str {
//             Box::leak(self.description.clone().into_boxed_str())
//         }

//         fn args_schema(&self) -> Value {
//             serde_json::json!({
//                 "type": "object",
//                 "properties": {
//                     "input": {"type": "string"}
//                 }
//             })
//         }

//         fn run(&self, args: Value) -> Value {
//             serde_json::json!({
//                 "result": format!("Executed {} with args: {}", self.name, args)
//             })
//         }
//     }

//     // Mock executor for testing
//     #[derive(Clone)]
//     struct MockExecutor {
//         system_prompt: String,
//         tools: Vec<Arc<dyn ToolT>>,
//     }

//     #[derive(Debug, Clone, Serialize, Deserialize)]
//     struct MockOutput {
//         message: String,
//     }

//     #[derive(Debug, thiserror::Error)]
//     enum MockError {
//         #[error("Mock error: {0}")]
//         Error(String),
//     }

//     #[async_trait]
//     impl AgentExecutor for MockExecutor {
//         type Output = MockOutput;
//         type Error = MockError;

//         async fn execute<L: LLMProvider + Clone + 'static>(
//             &self,
//             _llm: &L,
//             _session: &mut Session,
//             _task: Task,
//         ) -> Result<Self::Output, Self::Error> {
//             Ok(MockOutput {
//                 message: "Test execution completed".to_string(),
//             })
//         }

//         async fn process_turn<L: LLMProvider + Clone + 'static>(
//             &self,
//             _llm: &L,
//             _session: &mut Session,
//             _messages: &mut Vec<ChatMessage>,
//         ) -> Result<TurnResult<Self::Output>, Self::Error> {
//             Ok(TurnResult::Complete(MockOutput {
//                 message: "Test turn completed".to_string(),
//             }))
//         }
//     }

//     #[test]
//     fn test_base_agent_creation() {
//         let executor = MockExecutor {
//             system_prompt: "Test prompt".to_string(),
//             tools: vec![],
//         };

//         let agent = BaseAgent::new(
//             "TestAgent".to_string(),
//             "Test description".to_string(),
//             executor,
//             vec![],
//         );

//         assert_eq!(agent.name, "TestAgent");
//         assert_eq!(agent.description, "Test description");
//         assert_eq!(agent.tools.len(), 0);
//     }

//     #[test]
//     fn test_base_agent_with_tools() {
//         let executor = MockExecutor {
//             system_prompt: "Test prompt".to_string(),
//             tools: vec![],
//         };

//         let tool1 = Arc::new(MockTool {
//             name: "tool1".to_string(),
//             description: "First tool".to_string(),
//         });

//         let tool2 = Arc::new(MockTool {
//             name: "tool2".to_string(),
//             description: "Second tool".to_string(),
//         });

//         let agent = BaseAgent::new(
//             "TestAgent".to_string(),
//             "Test description".to_string(),
//             executor,
//             vec![],
//         )
//         .with_tool(tool1.clone())
//         .with_tool(tool2.clone());

//         assert_eq!(agent.tools.len(), 2);
//         assert_eq!(agent.get_all_tools().len(), 2);
//     }

//     #[test]
//     fn test_executor_with_tools() {
//         let tool = Arc::new(MockTool {
//             name: "executor_tool".to_string(),
//             description: "Executor tool".to_string(),
//         });

//         let executor = MockExecutor {
//             system_prompt: "Test prompt".to_string(),
//             tools: vec![tool.clone()],
//         };

//         let agent_tool = Arc::new(MockTool {
//             name: "agent_tool".to_string(),
//             description: "Agent tool".to_string(),
//         });

//         let agent = BaseAgent::new(
//             "TestAgent".to_string(),
//             "Test description".to_string(),
//             executor,
//             vec![],
//         )
//         .with_tool(agent_tool);

//         // Should have both executor and agent tools
//         assert_eq!(agent.get_all_tools().len(), 2);
//     }

//     #[test]
//     fn test_tool_wrapper() {
//         let tool = Arc::new(MockTool {
//             name: "test_tool".to_string(),
//             description: "Test tool".to_string(),
//         });

//         let wrapper = ToolWrapper::new(tool.clone());

//         assert_eq!(wrapper.name(), "test_tool");
//         assert_eq!(wrapper.description(), "Test tool");

//         let args = serde_json::json!({"input": "test"});
//         let result = wrapper.run(args.clone());

//         assert!(result.to_string().contains("Executed test_tool"));
//     }
// }
