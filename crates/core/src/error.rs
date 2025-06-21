use uuid::Uuid;

/// Error types for Environment operations
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
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
