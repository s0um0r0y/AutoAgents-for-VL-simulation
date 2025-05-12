use super::transport::Transport;
use crate::agent::{Agent, AgentT};
use async_trait::async_trait;

#[async_trait]
pub trait AgentRuntime<'a>: Sized {
    type TRANSPORT: Transport;
    async fn send_message(&self, message: String, recipient: String, sender: String) -> String;
    async fn publish_message(&self, message: String, topic_id: String, sender: String) -> String;
    async fn start(&mut self);
    fn register_agent(&mut self, agent: impl AgentT + 'a);
}
