use autoagents::agent::base::AgentDeriveT;
use autoagents::agent::types::SimpleAgentBuilder;
use autoagents::environment::Environment;
use autoagents::tool::Tool;
use autoagents::tool::ToolInputT;
use autoagents_derive::{agent, tool, ToolInput};
use autoagents_llm::llm::LLM;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct WeatherArgs {}

#[tool(
    name = "WeatherTool",
    description = "Use this tool to get current weather in celcius",
    input = WeatherArgs,
    output = String
)]
fn get_weather(args: WeatherArgs) -> String {
    println!("GetWeather {:?}", args);
    "23 degree".into()
}

#[agent(
    name = "general_agent",
    description = "You are general assistant and will answer user quesries in crips manner.",
    tools = [WeatherTool]
)]
pub struct MathAgent {}

pub async fn react_agent(llm: impl LLM + Clone + 'static) {
    // Build a Simple agent with the provided system prompt
    let agent =
        SimpleAgentBuilder::new("general_agent", "You are a helpful assistant.".to_string())
            .description("General assistant that answers user queries in a crisp manner")
            .build();

    // Set up a task
    let task = "What is the current weather??";

    let mut environment = Environment::new(llm, None);
    let agent_id = environment.register_agent(agent, None);
    let _task = environment.add_task(agent_id, task).await.unwrap();
    let result = environment.run(agent_id, None).await.unwrap();
    println!("{:?}", result);
}
