use autoagents_llm::providers::ollama::{model::OllamaModel, Ollama};
use autoagents_llm::providers::openai::{model::OpenAIModel, OpenAI};
use autoagents_llm::llm::{ChatMessage, ChatRole, TextGenerationOptions, ChatCompletionOptions, LLM};
use clap::{Parser, ValueEnum};
use std::io::{self, Write};

#[derive(Debug, Clone, ValueEnum)]
enum Provider {
    Ollama,
    OpenAI,
    Both,
}

#[derive(Debug, Clone, ValueEnum)]
enum Mode {
    Chat,
    TextGen,
}

#[derive(Parser, Debug)]
#[command(version, about = "Compare different LLM providers", long_about = None)]
struct Args {
    #[arg(short, long, value_enum)]
    provider: Provider,
    
    #[arg(short, long, help = "Your message to the LLM")]
    message: String,
    
    #[arg(long, help = "Stream the response")]
    stream: bool,
    
    #[arg(long, value_enum, default_value = "chat", help = "Mode of operation")]
    mode: Mode,
    
    #[arg(long, default_value = "512", help = "Maximum tokens to generate")]
    max_tokens: i32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    
    match args.provider {
        Provider::Ollama => {
            println!("🤖 Using Ollama provider (Local)...");
            println!("📋 Model: Qwen2.5 14B");
            let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_14B);
            run_with_ollama(llm, &args).await?;
        }
        Provider::OpenAI => {
            println!("🌐 Using OpenAI provider (Cloud)...");
            println!("📋 Model: GPT-3.5 Turbo");
            
            // Check if API key is set
            if std::env::var("OPENAI_API_KEY").is_err() {
                eprintln!("❌ Error: OPENAI_API_KEY environment variable not set");
                eprintln!("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here");
                std::process::exit(1);
            }
            
            let llm = OpenAI::new().set_model(OpenAIModel::GPT35Turbo);
            run_with_openai(llm, &args).await?;
        }
        Provider::Both => {
            println!("🔄 Comparing both providers...");
            
            // Check if API key is set for OpenAI
            if std::env::var("OPENAI_API_KEY").is_err() {
                eprintln!("❌ Error: OPENAI_API_KEY environment variable not set");
                eprintln!("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here");
                std::process::exit(1);
            }
            
            println!("\n--- Ollama Response ---");
            let ollama_llm = Ollama::new().set_model(OllamaModel::Qwen2_5_14B);
            run_with_ollama(ollama_llm, &args).await?;
            
            println!("\n--- OpenAI Response ---");
            let openai_llm = OpenAI::new().set_model(OpenAIModel::GPT35Turbo);
            run_with_openai(openai_llm, &args).await?;
        }
    }
    
    Ok(())
}

async fn run_with_openai(
    llm: OpenAI,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    
    match args.mode {
        Mode::Chat => chat_with_openai(llm, &args.message, args.stream, args.max_tokens).await?,
        Mode::TextGen => text_gen_with_openai(llm, &args.message, args.stream, args.max_tokens).await?,
    }
    
    let duration = start.elapsed();
    println!("⏱️  OpenAI response time: {:.2}s", duration.as_secs_f64());
    Ok(())
}

async fn run_with_ollama(
    llm: Ollama,
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    
    match args.mode {
        Mode::Chat => chat_with_ollama(llm, &args.message, args.stream, args.max_tokens).await?,
        Mode::TextGen => text_gen_with_ollama(llm, &args.message, args.stream, args.max_tokens).await?,
    }
    
    let duration = start.elapsed();
    println!("⏱️  Ollama response time: {:.2}s", duration.as_secs_f64());
    Ok(())
}

async fn chat_with_openai(
    llm: OpenAI,
    message: &str,
    stream: bool,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: "You are a helpful assistant. Be concise but informative.".to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: message.to_string(),
        },
    ];
    
    let options = Some(ChatCompletionOptions { num_tokens: max_tokens });
    
    println!("💬 You: {}", message);
    print!("🤖 Assistant: ");
    io::stdout().flush().unwrap();
    
    if stream {
        use futures::StreamExt;
        let mut stream = llm.chat_completion_stream(messages, options).await;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    print!("{}", response.message.content);
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                    break;
                }
            }
        }
        println!(); // New line after streaming
    } else {
        match llm.chat_completion(messages, options).await {
            Ok(response) => {
                println!("{}", response.message.content);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn chat_with_ollama(
    llm: Ollama,
    message: &str,
    stream: bool,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let messages = vec![
        ChatMessage {
            role: ChatRole::System,
            content: "You are a helpful assistant. Be concise but informative.".to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            content: message.to_string(),
        },
    ];
    
    let options = Some(ChatCompletionOptions { num_tokens: max_tokens });
    
    println!("💬 You: {}", message);
    print!("🤖 Assistant: ");
    io::stdout().flush().unwrap();
    
    if stream {
        use futures::StreamExt;
        let mut stream = llm.chat_completion_stream(messages, options).await;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    print!("{}", response.message.content);
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                    break;
                }
            }
        }
        println!(); // New line after streaming
    } else {
        match llm.chat_completion(messages, options).await {
            Ok(response) => {
                println!("{}", response.message.content);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn text_gen_with_openai(
    llm: OpenAI,
    prompt: &str,
    stream: bool,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let system_prompt = "You are a creative writing assistant.";
    let options = Some(TextGenerationOptions { num_tokens: max_tokens });
    
    println!("📝 Prompt: {}", prompt);
    print!("✨ Generated: ");
    io::stdout().flush().unwrap();
    
    if stream {
        use futures::StreamExt;
        let mut stream = llm.text_generation_stream(prompt, system_prompt, options).await;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    print!("{}", response.response);
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                    break;
                }
            }
        }
        println!(); // New line after streaming
    } else {
        match llm.text_generation(prompt, system_prompt, options).await {
            Ok(response) => {
                println!("{}", response.response);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn text_gen_with_ollama(
    llm: Ollama,
    prompt: &str,
    stream: bool,
    max_tokens: i32,
) -> Result<(), Box<dyn std::error::Error>> {
    let system_prompt = "You are a creative writing assistant.";
    let options = Some(TextGenerationOptions { num_tokens: max_tokens });
    
    println!("📝 Prompt: {}", prompt);
    print!("✨ Generated: ");
    io::stdout().flush().unwrap();
    
    if stream {
        use futures::StreamExt;
        let mut stream = llm.text_generation_stream(prompt, system_prompt, options).await;
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(response) => {
                    print!("{}", response.response);
                    io::stdout().flush().unwrap();
                }
                Err(e) => {
                    eprintln!("❌ Error: {}", e);
                    break;
                }
            }
        }
        println!(); // New line after streaming
    } else {
        match llm.text_generation(prompt, system_prompt, options).await {
            Ok(response) => {
                println!("{}", response.response);
            }
            Err(e) => {
                eprintln!("❌ Error: {}", e);
            }
        }
    }
    
    Ok(())
}