use super::{message::ChannelMessage, AgentRuntime};
use crate::{agent::Agent, context::LLMContext, message::Message};
use tokio::sync::mpsc;

pub struct SingleThreadedAgentRuntime<'a> {
    context: LLMContext,
    registered_agents: Vec<Box<dyn Agent + 'a>>,
    receiver: mpsc::Receiver<ChannelMessage>,
    sender: mpsc::Sender<ChannelMessage>,
}

impl<'a> SingleThreadedAgentRuntime<'a> {
    pub fn new(config: crate::llm::LLMConfig) -> Self {
        let (tx, rx) = mpsc::channel(100);
        Self {
            context: LLMContext::new(config),
            registered_agents: vec![],
            receiver: rx,
            sender: tx,
        }
    }

    pub fn get_channel(&self) -> mpsc::Sender<ChannelMessage> {
        self.sender.clone()
    }
}

impl<'a> AgentRuntime<'a> for SingleThreadedAgentRuntime<'a> {
    async fn send_message(
        &self,
        message: impl Message,
        recipient: String,
        sender: String,
    ) -> String {
        todo!()
    }

    async fn publish_message(&self, message: String, topic_id: String, sender: String) -> String {
        todo!()
    }

    async fn start(&mut self) {
        println!("Runtime started!");

        while let Some(msg) = self.receiver.recv().await {
            if let Some(agent) = self
                .registered_agents
                .iter()
                .find(|a| a.id() == msg.agent_id)
            {
                agent.on_message(msg.agent_id.clone(), "test".into()).await;
            } else {
                println!("Agent with ID {} not found!", msg.agent_id);
            }
        }

        println!("Runtime shutting down...");
    }

    fn register_agent(&mut self, agent: impl Agent + 'a) {
        self.registered_agents.push(Box::new(agent));
    }
}
