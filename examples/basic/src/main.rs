use autoagents::providers::ollama::{model::OllamaModel, Ollama};
use clap::{Parser, ValueEnum};

mod chat;
mod text_gen;
mod tool;

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    Chat,
    Tool,
    TextGen,
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
    let llm = Ollama::new().with_model(OllamaModel::Qwen2Coder7B);
    match args.usecase {
        UseCase::Tool => tool::tool(llm).await,
        UseCase::Chat => chat::stream(llm).await,
        UseCase::TextGen => text_gen::text_gen(llm).await,
    }
}
