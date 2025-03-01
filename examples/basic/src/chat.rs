#![allow(dead_code, unused_imports)]
use std::{alloc::System, time::SystemTime};

use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{tool, ToolInput};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput)]
pub struct GetCurrentWeatherArgs {}

#[tool(
    name = "GetCurrentWeather",
    description = "Use this tool to get the current Weather",
    input = GetCurrentWeatherArgs,
    output = String
)]
fn get_current_weather(_args: GetCurrentWeatherArgs) -> String {
    // Dummy implementation.
    format!("Current Time is {:?}", SystemTime::now())
}

pub async fn stream(mut llm: impl LLM) {
    llm.register_tool(GetCurrentWeather);
    let mut stream = llm
        .chat_completion_stream(
            vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are an Assistant and your name is Lilly".into(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hello!".into(),
                },
            ],
            None,
        )
        .await;

    // Iterate over the stream as chunks arrive.
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => {
                // Here, `response` is of type TextGenerationResponse.
                println!("Received chunk: {}", response);
            }
            Err(e) => {
                eprintln!("Error in streaming response: {}", e);
            }
        }
    }
}
