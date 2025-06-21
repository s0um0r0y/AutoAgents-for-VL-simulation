use autoagents::agent::base::AgentDeriveT;
use autoagents::agent::types::SimpleAgentBuilder;
use autoagents::environment::Environment;
use autoagents::protocol::Event;
use autoagents::{LLMProvider, ToolInputT, ToolT};
use autoagents_derive::{agent, tool, ToolInput};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::mpsc::Receiver;

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
pub struct WeatherAgent {}

fn handle_events(mut event_receiver: Receiver<Event>) {
    tokio::spawn(async move {
        while let Some(event) = event_receiver.recv().await {
            println!("Event: {:?}", event);
        }
    });
}

pub async fn simple_agent(llm: Arc<Box<dyn LLMProvider>>) {
    // Build a Simple agent
    let agent = SimpleAgentBuilder::from_agent(
        WeatherAgent {},
        "You are general assistant and will answer user quesries in crisp manner.".into(),
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
