use autoagents::{
    providers::ollama::{model::OllamaModel, Ollama},
    runtime::{AgentRuntime, SingleThreadedAgentRuntime},
};
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum)]
enum UseCase {
    SingleThreaded,
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
    let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_14B);
    match args.usecase {
        UseCase::SingleThreaded => {
            let mut runtime = SingleThreadedAgentRuntime::new("Hello".into());
            runtime.start().await;
        }
    }
}
