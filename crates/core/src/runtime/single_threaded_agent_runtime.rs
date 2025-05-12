use super::{context::LLMContext, message::ChannelMessage, transport::Transport, AgentRuntime};
use crate::agent::{Agent, AgentT};
use async_trait::async_trait;
use tokio::sync::mpsc;

pub struct SingleThreadedAgentRuntime {
    context: LLMContext,
    registered_agents: Vec<String>,
    transport: MPCTransport,
}

impl SingleThreadedAgentRuntime {
    pub fn new(config: String) -> Self {
        let (tx, rx) = mpsc::channel(100);
        Self {
            context: LLMContext::new(config),
            registered_agents: vec![],
            transport: MPCTransport {
                receiver: rx,
                sender: tx,
            },
        }
    }

    pub fn get_channel(&self) -> mpsc::Sender<ChannelMessage> {
        // self.sender.clone()
        todo!()
    }
}

pub struct MPCTransport {
    receiver: mpsc::Receiver<ChannelMessage>,
    sender: mpsc::Sender<ChannelMessage>,
}

#[async_trait]
impl Transport for MPCTransport {
    async fn send(message: String) {
        //
    }
}

#[async_trait]
impl<'a> AgentRuntime<'a> for SingleThreadedAgentRuntime {
    type TRANSPORT = MPCTransport;

    async fn send_message(&self, message: String, recipient: String, sender: String) -> String {
        todo!()
    }

    async fn publish_message(&self, message: String, topic_id: String, sender: String) -> String {
        todo!()
    }

    async fn start(&mut self) {
        println!("Runtime started!");

        // while let Some(msg) = self.receiver.recv().await {
        //     if let Some(agent) = self
        //         .registered_agents
        //         .iter()
        //         .find(|a| a.id() == msg.agent_id)
        //     {
        //         agent.on_message(msg.agent_id.clone(), "test".into()).await;
        //     } else {
        //         println!("Agent with ID {} not found!", msg.agent_id);
        //     }
        // }

        println!("Runtime shutting down...");
    }

    fn register_agent(&mut self, agent: impl AgentT + 'a) {
        // self.registered_agents.push(Box::new(agent));
    }
}
