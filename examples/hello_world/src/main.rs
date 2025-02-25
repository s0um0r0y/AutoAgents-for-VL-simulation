use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::Ollama,
};

#[tokio::main]
async fn main() {
    let llm = Ollama::new();
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
}
