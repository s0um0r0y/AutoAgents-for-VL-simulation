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
pub struct GetCurrentWeatherArgs {
    #[input(description = "The location to get the weather for, e.g. San Francisco, CA")]
    pub location: String,
    #[input(
        description = "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
        choice = ["celsius", "fahrenheit"]
    )]
    pub format: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GetCurrentWeatherOutput {
    pub weather: String,
}

#[tool(
    name = "GetCurrentWeather",
    description = "Get the current weather for a location, use this only when requested for weather",
    input = GetCurrentWeatherArgs,
    output = GetCurrentWeatherOutput
)]
fn get_current_weather(args: GetCurrentWeatherArgs) -> GetCurrentWeatherOutput {
    // Dummy implementation.
    GetCurrentWeatherOutput {
        weather: format!("Weather in {} is sunny (in {})", args.location, args.format),
    }
}

pub async fn tool(mut llm: impl LLM) {
    llm.register_tool(GetCurrentWeather);

    let tool_resul = llm.call_tool(
        "GetCurrentWeather",
        serde_json::from_str("{\"location\":\"test\",\"format\":\"tst\"}").unwrap(),
    );
    println!("Tool Result {:?}", tool_resul);

    let response = llm
        .chat_completion(
            vec![
                ChatMessage {
                    role: ChatRole::System,
                    content: "You are an Assistant who helps resolve user queries. Use tools only when asked for!".into(),
                },
                ChatMessage {
                    role: ChatRole::User,
                    content: "Hey, What is the current weather in Seattle?".into(),
                },
            ],
            None,
        )
        .await
        .unwrap();

    match response.message.role {
        ChatRole::Assistant => {
            let tools_call = response.message.tool_calls;
            for tool in tools_call {
                let tool_func = &tool.function;
                let tool_resul = llm.call_tool(&tool_func.name, tool_func.arguments.clone());
                println!("Tool Result: {:?}", tool_resul);
            }
        }
        _ => {
            //Ignore
        }
    }
}
