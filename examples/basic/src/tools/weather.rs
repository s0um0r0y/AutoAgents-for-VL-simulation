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
