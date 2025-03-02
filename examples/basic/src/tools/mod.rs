#![allow(dead_code, unused_imports)]
use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{tool, ToolInput};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::chat::GetCurrentWeather;

pub mod news;
pub mod summarize;

pub async fn run_tool(mut llm: impl LLM) {
    llm.register_tool(Box::new(GetCurrentWeather));

    let tool_resul = llm.call_tool(
        "GetCurrentWeather",
        serde_json::from_str("{\"location\":\"test\",\"format\":\"tst\"}").unwrap(),
    );
    println!("Tool Result {:?}", tool_resul);

    let response = llm
        .chat_completion(
            vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are an Assistant who helps resolve user queries. Use tools only when asked for!".into(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hey, What is the current weather in Seattle?".into(),
                },
            ],
            None,
        )
        .await
        .unwrap();

    match response.message.role {
        ChatRole::Assistant => {
            let tools_call = response.message.tool_calls;
            for tool in tools_call {
                let tool_func = &tool.function;
                let tool_resul = llm.call_tool(&tool_func.name, tool_func.arguments.clone());
                println!("Tool Result: {:?}", tool_resul);
            }
        }
        _ => {
            //Ignore
        }
    }
}
