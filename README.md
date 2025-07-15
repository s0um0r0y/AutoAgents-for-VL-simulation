# AutoAgents 🚀
A multi-agent framework powered by Rust and modern LLMs.


🎥 **Watch the full demo on YouTube** to see AutoAgents in action!

[![Coding Agent Demo](https://img.youtube.com/vi/MZLd4aRuftM/maxresdefault.jpg)](https://youtu.be/MZLd4aRuftM?si=XbHitYjgyffyOf5D)

---

## Features

- **Multi-Agent Coordination:** Allow agents to interact, share knowledge, and collaborate on complex tasks.
- **Tool Integration:** Seamlessly integrate with external tools and services to extend agent capabilities.
- **Extensible Framework:** Easily add new providers and tools using our intuitive plugin system.
- **Structured Agent Output:** Easily add json schema for agent Output
---

## Supported Providers

**multiple LLM backends** in a single project: [OpenAI](https://openai.com), [Anthropic (Claude)](https://www.anthropic.com), [Ollama](https://github.com/ollama/ollama), [DeepSeek](https://www.deepseek.com), [xAI](https://x.ai), [Phind](https://www.phind.com), [Groq](https://www.groq.com), [Google](https://cloud.google.com/gemini).

*Note: Provider support is actively evolving.*


## Instructions to build
```sh
git clone https://github.com/liquidos-ai/autoagents.git
cd autoagents
cargo build --release
```

## Usage

### LLM Tool Calling
```rs
use autoagents::core::agent::base::AgentBuilder;
use autoagents::core::agent::output::AgentOutputT;
use autoagents::core::agent::{AgentDeriveT, ReActAgentOutput, ReActExecutor};
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::llm::{LLMProvider, ToolCallError, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, AgentOutput, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AdditionArgs {
    #[input(description = "Left Operand for addition")]
    left: i64,
    #[input(description = "Right Operand for addition")]
    right: i64,
}

#[tool(
    name = "Addition",
    description = "Use this tool to Add two numbers",
    input = AdditionArgs,
)]
fn add(args: AdditionArgs) -> Result<i64, ToolCallError> {
    println!("ToolCall: Add {:?}", args);
    Ok(args.left + args.right)
}

/// Math agent output with Value and Explanation
#[derive(Debug, Serialize, Deserialize, AgentOutput)]
pub struct MathAgentOutput {
    #[output(description = "The addition result")]
    value: i64,
    #[output(description = "Explanation of the logic")]
    explanation: String,
}

#[agent(
    name = "math_agent",
    description = "You are a Math agent.",
    tools = [Addition],
    executor = ReActExecutor,
    output = MathAgentOutput
)]
pub struct MathAgent {}

pub async fn math_agent(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("Starting Math Agent Example...\n");

    let sliding_window_memory = Box::new(SlidingWindowMemory::new(10));

    let agent = MathAgent {};

    let agent = AgentBuilder::new(agent)
        .with_llm(llm)
        .with_memory(sliding_window_memory)
        .build()?;

    // Create environment and set up event handling
    let mut environment = Environment::new(None).await;

    // Register the agent
    let agent_id = environment.register_agent(agent.clone(), None).await?;

    // Add tasks
    let _ = environment.add_task(agent_id, "What is 5 + 3?").await?;

    // Run all tasks
    let results = environment.run_all(agent_id, None).await?;
    let last_result = results.last().unwrap();
    let agent_output: MathAgentOutput =
        last_result.extract_agent_output(ReActAgentOutput::extract_agent_output)?;
    println!(
        "Results: {}, {}",
        agent_output.value, agent_output.explanation
    );

    // Shutdown
    let _ = environment.shutdown().await;
    Ok(())
}
```


## Contributing
We welcome contributions!

## License
MIT OR Apache-2.0. See [MIT_LICENSE](MIT_LICENSE) or [APACHE_LICENSE](APACHE_LICENSE) for more.
