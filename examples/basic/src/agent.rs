#![allow(dead_code, unused_imports)]
use crate::tools::{news::SearchNews, summarize::SummarizeContent};
use async_trait::async_trait;
use autoagents::{
    agent::{types::ReActAgentT, Agent, AgentDeriveT, AgentT},
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{agent, tool, ToolInput};
use futures::{stream::StreamExt, task::waker};
use serde::{Deserialize, Serialize};
use serde_json::{Error, Value};
use std::{alloc::System, time::SystemTime};

#[derive(Serialize, Deserialize, PartialEq, Eq, Debug)]
struct AgentOutput {}

#[agent(
    name = "test_agent",
    description = "An AI research assistant skilled in summarizing News Articles. Your goal is to gather and summarize recent News Articles using a ReAct approach."
    tools = [SearchNews, SummarizeContent])
]
pub struct TestAgent {}

#[async_trait]
impl AgentT for TestAgent {
    type Err = Error;
    type Output = String;

    async fn call<T: LLM>(&self, llm: &mut T, prompt: &str) -> Result<Self::Output, Self::Err> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.into(),
        }];
        let response = self.chat_completion(llm, messages).await.unwrap();
        return Ok(response.to_string());
    }
}

#[async_trait]
impl ReActAgentT for TestAgent {
    async fn step<L: LLM + 'static>(
        &self,
        llm: &mut L,
        messages: Vec<ChatMessage>,
    ) -> Result<String, Self::Err> {
        // Each step uses the agent's chat_completion method to produce a new chain-of-thought iteration.
        let response = self.chat_completion(llm, messages).await.unwrap();
        Ok(response.to_string())
    }
}

pub async fn agent(mut llm: impl LLM + 'static) {
    let mut agent = Agent::new(TestAgent {}, &mut llm).unwrap();
    println!("Agent Name: {}", agent.name());
    println!("Agent Description: {}", agent.description());
    let response = agent
        .run_react("What is latest summarized news on AI Research")
        .await
        .unwrap();
    println!("Response: {}", response);
}
