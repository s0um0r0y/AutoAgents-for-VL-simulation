use super::error::AgentError;
use super::executor::AgentExecutor;
use async_trait::async_trait;
use autoagents_llm::llm::LLM;
use autoagents_llm::tool::Tool;
use serde_json::Value;
use std::fmt::Debug;
use std::sync::Arc;

/// Base agent type that uses an executor strategy
#[derive(Clone)]
pub struct BaseAgent<E: AgentExecutor + Send + Sync> {
    pub name: String,
    pub description: String,
    pub executor: E,
    pub tools: Vec<Arc<dyn Tool>>,
}

impl<E: AgentExecutor> BaseAgent<E> {
    pub fn new(name: String, description: String, executor: E, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            name,
            description,
            executor,
            tools,
        }
    }

    /// Add a tool to this agent
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add multiple tools to this agent
    pub fn with_tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Get all tools from both the agent and executor
    fn get_all_tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }

    /// Register all tools with the LLM
    pub fn register_tools<L: LLM>(&self, llm: &mut L) -> Result<(), AgentError> {
        let all_tools = self.get_all_tools();

        if !all_tools.is_empty() && !llm.supports_tools() {
            return Err(AgentError::ModelToolNotSupported(
                llm.model_name().to_string(),
            ));
        }

        for tool in all_tools {
            // Create a wrapper that implements Tool and can be boxed
            let tool_wrapper = ToolWrapper::new(tool);
            llm.register_tool(Box::new(tool_wrapper));
        }

        Ok(())
    }
}

#[async_trait]
pub trait AgentDeriveT: Send + Sync {
    fn description(&self) -> &'static str;
    fn name(&self) -> &'static str;
    fn tools(&self) -> Vec<Box<dyn Tool>>;
}

/// A wrapper that allows Arc<dyn Tool> to be used as Box<dyn Tool>
#[derive(Clone)]
pub(crate) struct ToolWrapper {
    tool: Arc<dyn Tool>,
}

impl ToolWrapper {
    pub fn new(tool: Arc<dyn Tool>) -> Self {
        Self { tool }
    }
}

impl Debug for ToolWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ToolWrapper({})", self.tool.name())
    }
}

impl Tool for ToolWrapper {
    fn name(&self) -> &'static str {
        self.tool.name()
    }

    fn description(&self) -> &'static str {
        self.tool.description()
    }

    fn args_schema(&self) -> Value {
        self.tool.args_schema()
    }

    fn run(&self, args: Value) -> Value {
        self.tool.run(args)
    }
}

#[cfg(test)]
mod tests {
    #![allow(dead_code)]
    use super::*;
    use crate::session::Session;
    use crate::{agent::executor::TurnResult, session::Task};
    use async_trait::async_trait;
    use autoagents_llm::llm::{ChatMessage, LLM};
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::sync::Arc;

    // Mock tool for testing
    #[derive(Debug, Clone)]
    struct MockTool {
        name: String,
        description: String,
    }

    impl Tool for MockTool {
        fn name(&self) -> &'static str {
            Box::leak(self.name.clone().into_boxed_str())
        }

        fn description(&self) -> &'static str {
            Box::leak(self.description.clone().into_boxed_str())
        }

        fn args_schema(&self) -> Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                }
            })
        }

        fn run(&self, args: Value) -> Value {
            serde_json::json!({
                "result": format!("Executed {} with args: {}", self.name, args)
            })
        }
    }

    // Mock executor for testing
    #[derive(Clone)]
    struct MockExecutor {
        system_prompt: String,
        tools: Vec<Arc<dyn Tool>>,
    }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct MockOutput {
        message: String,
    }

    #[derive(Debug, thiserror::Error)]
    enum MockError {
        #[error("Mock error: {0}")]
        Error(String),
    }

    #[async_trait]
    impl AgentExecutor for MockExecutor {
        type Output = MockOutput;
        type Error = MockError;

        async fn execute<L: LLM + Clone + 'static>(
            &self,
            _llm: &L,
            _session: &mut Session,
            _task: Task,
        ) -> Result<Self::Output, Self::Error> {
            Ok(MockOutput {
                message: "Test execution completed".to_string(),
            })
        }

        async fn process_turn<L: LLM + Clone + 'static>(
            &self,
            _llm: &L,
            _session: &mut Session,
            _messages: &mut Vec<ChatMessage>,
        ) -> Result<TurnResult<Self::Output>, Self::Error> {
            Ok(TurnResult::Complete(MockOutput {
                message: "Test turn completed".to_string(),
            }))
        }
    }

    #[test]
    fn test_base_agent_creation() {
        let executor = MockExecutor {
            system_prompt: "Test prompt".to_string(),
            tools: vec![],
        };

        let agent = BaseAgent::new(
            "TestAgent".to_string(),
            "Test description".to_string(),
            executor,
            vec![],
        );

        assert_eq!(agent.name, "TestAgent");
        assert_eq!(agent.description, "Test description");
        assert_eq!(agent.tools.len(), 0);
    }

    #[test]
    fn test_base_agent_with_tools() {
        let executor = MockExecutor {
            system_prompt: "Test prompt".to_string(),
            tools: vec![],
        };

        let tool1 = Arc::new(MockTool {
            name: "tool1".to_string(),
            description: "First tool".to_string(),
        });

        let tool2 = Arc::new(MockTool {
            name: "tool2".to_string(),
            description: "Second tool".to_string(),
        });

        let agent = BaseAgent::new(
            "TestAgent".to_string(),
            "Test description".to_string(),
            executor,
            vec![],
        )
        .with_tool(tool1.clone())
        .with_tool(tool2.clone());

        assert_eq!(agent.tools.len(), 2);
        assert_eq!(agent.get_all_tools().len(), 2);
    }

    #[test]
    fn test_executor_with_tools() {
        let tool = Arc::new(MockTool {
            name: "executor_tool".to_string(),
            description: "Executor tool".to_string(),
        });

        let executor = MockExecutor {
            system_prompt: "Test prompt".to_string(),
            tools: vec![tool.clone()],
        };

        let agent_tool = Arc::new(MockTool {
            name: "agent_tool".to_string(),
            description: "Agent tool".to_string(),
        });

        let agent = BaseAgent::new(
            "TestAgent".to_string(),
            "Test description".to_string(),
            executor,
            vec![],
        )
        .with_tool(agent_tool);

        // Should have both executor and agent tools
        assert_eq!(agent.get_all_tools().len(), 2);
    }

    #[test]
    fn test_tool_wrapper() {
        let tool = Arc::new(MockTool {
            name: "test_tool".to_string(),
            description: "Test tool".to_string(),
        });

        let wrapper = ToolWrapper::new(tool.clone());

        assert_eq!(wrapper.name(), "test_tool");
        assert_eq!(wrapper.description(), "Test tool");

        let args = serde_json::json!({"input": "test"});
        let result = wrapper.run(args.clone());

        assert!(result.to_string().contains("Executed test_tool"));
    }
}
