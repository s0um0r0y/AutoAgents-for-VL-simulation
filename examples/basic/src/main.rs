use autoagents::providers::{
    ollama::{model::OllamaModel, Ollama},
    openai::{model::OpenAIModel, OpenAI},
};
use clap::{Parser, ValueEnum};
mod chat;
mod react;
mod text_gen;
mod tools;
use tools::run_tool;
mod simple;

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    OpenAI,
    Ollama,
}

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Chat,
    Tool,
    Simple,
    TextGen,
    React,
}

/// Simple program to demonstrate AutoAgents functionality
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, help = "Use case to run")]
    usecase: UseCase,
    #[arg(short, long, help = "LLM Model")]
    model: Model,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    // Check if API key is set
    if std::env::var("OPENAI_API_KEY").is_err() {
        eprintln!("❌ Error: OPENAI_API_KEY environment variable not set");
        eprintln!("💡 Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here");
        std::process::exit(1);
    }

    match args.model {
        Model::OpenAI => {
            let llm = OpenAI::new().set_model(OpenAIModel::GPT4);
            match args.usecase {
                UseCase::Tool => run_tool(llm).await,
                UseCase::Chat => chat::stream(llm).await,
                UseCase::TextGen => text_gen::text_gen(llm).await,
                UseCase::React => react::react_agent(llm).await,
                UseCase::Simple => simple::simple_agent(llm).await,
            }
        }
        Model::Ollama => {
            let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_14B);
            match args.usecase {
                UseCase::Tool => run_tool(llm).await,
                UseCase::Chat => chat::stream(llm).await,
                UseCase::TextGen => text_gen::text_gen(llm).await,
                UseCase::React => react::react_agent(llm).await,
                UseCase::Simple => simple::simple_agent(llm).await,
            }
        }
    }
}
