#![allow(dead_code, unused_imports)]
use crate::tools::{news::SearchNews, summarize::SummarizeContent};
use async_trait::async_trait;
use autoagents::{
    agent::{types::ReActAgentT, Agent, AgentDeriveT, AgentT},
    llm::{ChatCompletionResponse, ChatMessage, ChatRole, TextGenerationOptions, LLM},
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
    tools = [SearchNews])
]
pub struct TestAgent {}

#[async_trait]
impl AgentT for TestAgent {
    type Err = Error;
    type Output = String;

    async fn call<T: LLM>(
        &self,
        llm: &mut T,
        prompt: &str,
    ) -> Result<ChatCompletionResponse, T::Error> {
        let messages = vec![ChatMessage {
            role: ChatRole::User,
            content: prompt.into(),
        }];
        self.chat_completion(llm, messages).await
    }
}

pub async fn agent(mut llm: impl LLM + 'static) {
    let mut agent = Agent::new(TestAgent {}, &mut llm).unwrap();
    println!("Agent Name: {}", agent.name());
    println!("Agent Description: {}", agent.description());
    let response = agent
        .run("What is latest news on AI Research")
        .await
        .unwrap();
    println!("Response: {}", response);
}
