#![allow(dead_code, unused_imports)]
use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::ToolArg,
    Tool,
};
use autoagents_derive::{tool, ToolArg};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub async fn stream(llm: impl LLM) {
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
