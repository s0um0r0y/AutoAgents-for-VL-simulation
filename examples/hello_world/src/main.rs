use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
};
use futures::stream::StreamExt;

#[tokio::main]
async fn main() {
    let llm = Ollama::new().with_model(OllamaModel::DeepSeekR18B);
    let response = llm
        .text_generation(
            "Why is the sky blue in 1 sentence?",
            "You are an assistant!",
            Some(TextGenerationOptions { num_tokens: 20 }),
        )
        .await
        .unwrap();
    println!("{}", response);

    let response = llm
        .chat_completion(
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
