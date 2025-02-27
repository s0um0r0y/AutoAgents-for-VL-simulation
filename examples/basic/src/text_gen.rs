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

pub async fn text_gen(llm: impl LLM) {
    let response = llm
        .text_generation(
            "Why is the sky blue in 1 sentence?",
            "You are an assistant!",
            Some(TextGenerationOptions { num_tokens: 20 }),
        )
        .await
        .unwrap();
    println!("{}", response);

    let mut stream = llm
        .text_generation_stream(
            "Why is the sky blue in 1 sentence?",
            "You are an assistant!",
            Some(TextGenerationOptions { num_tokens: 20 }),
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
