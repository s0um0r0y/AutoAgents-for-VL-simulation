# AutoAgents

**AutoAgents** is a distributed multi-agent framework written in Rust that enables you to build, deploy, and coordinate multiple intelligent agents. AutoAgents leverages modern language models and tool integrations to provide robust, scalable, and flexible solutions for various use cases—from natural language processing and automated chat to advanced tool orchestration.

---

## Features

- **Distributed Architecture:** Coordinate multiple agents across different nodes.
- **Multi-Agent Coordination:** Allow agents to interact, share knowledge, and collaborate on complex tasks.
- **Tool Integration:** Seamlessly integrate with external tools and services to extend agent capabilities.
- **Extensible Framework:** Easily add new providers and tools using our intuitive plugin system.

---

## Supported Providers

AutoAgents currently supports the **Ollama** provider. The following table summarizes the supported models:

| Provider | Supported Models               |
|----------|--------------------------------|
| Ollama   | Llama-3.2, DeepSeek-R1-8B      |

*Note: Provider support is actively evolving.*


## Instructions to build
```sh
git clone https://github.com/yourusername/autoagents.git
cd autoagents
cargo build --release
```

## Usage

### LLM Tool Calling
```rs
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

// Declarative way to define Tool Inputs and Tool
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

#[tokio::main]
async fn main() {
    let llm = Ollama::new().with_model(OllamaModel::Qwen2_5_32B);
    //Register the tool
    llm.register_tool(GetCurrentWeather);

    //Call the tool manually
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

    //Handle the Chat Completion for Tool Call
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
```

## TODO
- [x] [REFACTOR] Refactor the Derive crate with more robust implementation
- [x] [FEATURE] Add Agent with tools and Agent Description
- [ ] [FEATURE] Add Protocol for App interaction
- [ ] [FEATURE] Add ReAct Agent type
- [ ] [FEATURE] Async Tools
- [ ] [REFACTOR] Refactor the Provider implementation where, Provider needs to do as little implemetation as possible,
  i.e take up the heavy lifting
- [ ] [REFACTOR] Add OpenAI Request and Response Which Other providers can directly reuse as it is most widley used
- [ ] [FEATURE] Add a basic Single Threaded Actor Model
- [ ] [COVERAGE] Add tests
