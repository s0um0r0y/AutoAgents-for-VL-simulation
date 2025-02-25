use crate::{agent::Agent, message::Message};

pub trait AgentRuntime<'a>: Sized {
    fn send_message(
        &self,
        message: impl Message,
        recipient: String,
        sender: String,
    ) -> impl std::future::Future<Output = String> + Send;
    fn publish_message(
        &self,
        message: String,
        topic_id: String,
        sender: String,
    ) -> impl std::future::Future<Output = String> + Send;
    fn start(&mut self) -> impl std::future::Future<Output = ()>;
    fn register_agent(&mut self, agent: impl Agent + 'a);
}
