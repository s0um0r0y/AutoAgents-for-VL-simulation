#![allow(dead_code, unused_imports)]
use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{tool, ToolInput};
use futures::stream::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, Debug, ToolInput)]
pub struct SummarizeArgs {
    #[input(description = "The input to be summarized")]
    pub input: String,
    #[input(
        description = "The format for the data to be summarized",
        choice = ["markdown"]
    )]
    pub format: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SummarizeOutput {
    pub output: String,
}

#[tool(
    name = "SummarizeContent",
    description = "Tool to summarize the input given the format",
    input = SummarizeArgs,
    output = SummarizeOutput
)]
fn summarize(args: SummarizeArgs) -> SummarizeOutput {
    let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_32B);
    let resp = llm.chat_completion_sync(
        vec![
            ChatMessage {
                role: ChatRole::System,
                content: "You are an Assistant who can summarize given information given a format"
                    .into(),
            },
            ChatMessage {
                role: ChatRole::User,
                content: format!(
                    "Summarize the below data given the format: '{}': \n DATA: \n{}",
                    args.format, args.input
                ),
            },
        ],
        None,
    );
    let summarized_val = resp.unwrap().message.content.to_string();
    SummarizeOutput {
        output: summarized_val,
    }
}
