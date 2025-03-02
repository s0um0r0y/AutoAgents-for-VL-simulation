use autoagents::providers::ollama::{model::OllamaModel, Ollama};
use clap::{Parser, ValueEnum};
mod agent;
mod chat;
mod text_gen;
mod tools;
use tools::run_tool;

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Chat,
    Tool,
    TextGen,
    Agent,
}

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    usecase: UseCase,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let llm = Ollama::new().set_model(OllamaModel::DeepSeekR18B);
    match args.usecase {
        UseCase::Tool => run_tool(llm).await,
        UseCase::Chat => chat::stream(llm).await,
        UseCase::TextGen => text_gen::text_gen(llm).await,
        UseCase::Agent => agent::agent(llm).await,
    }
}
