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
| OpenAI   | Gpt-4, Gpt-4o                    |

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
#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {}

#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celcius",
    input = WeatherArgs,
    output = String
)]
fn get_weather(args: WeatherArgs) -> String {
    println!("ToolCall: GetWeather {:?}", args);
    "23 degree".into()
}

#[agent(
    name = "general_agent",
    description = "You are general assistant and will answer user quesries in crips manner.",
    tools = [WeatherTool]
)]
pub struct MathAgent {}

fn handle_events(mut event_receiver: Receiver<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("Event: {:?}", event);
        }
    });
}

pub async fn simple_agent(llm: impl LLM + Clone + 'static) {
    // Build a Simple agent
    let agent = SimpleAgentBuilder::from_agent(
        MathAgent {},
        "You are general assistant and will answer user quesries in crips manner.".into(),
    )
    .build();

    let mut environment = Environment::new(llm, None);
    let receiver = environment.event_receiver().unwrap();
    handle_events(receiver);

    let agent_id = environment.register_agent(agent, None);

    let _ = environment
        .add_task(agent_id, "What is the current weather??")
        .await
        .unwrap();
    let _ = environment
        .add_task(
            agent_id,
            "What is the weather you just said? reply back in beautiful format.",
        )
        .await
        .unwrap();
    let result = environment.run_all(agent_id, None).await.unwrap();
    println!("Result {:?}", result);
    let _ = environment.shutdown().await;
}
```

## TODO
- [ ] [FEATURE] Refactor LLM Crate to be similar to LiteLLM
- [ ] [FEATURE] Add Cost Tracking in LLM Crate
- [ ] [REFACTOR] Seperate out embeddings into embeddings provider
- [ ] [REFACTOR] Unified interface for LLMProvider
- [ ] [REFACTOR] Proper implemetation of streaming and HTTP requests
- [ ] [FEATURE] Add ReAct Agent type
- [ ] [REFACTOR] Refactor the Provider implementation where, Provider needs to do as little implemetation as possible,
  i.e take up the heavy lifting
- [ ] [COVERAGE] Add tests
