use crate::agent::CodingAgent;
use autoagents::core::agent::prebuilt::react::ReActAgentOutput;
use autoagents::core::agent::AgentBuilder;
use autoagents::core::environment::Environment;
use autoagents::core::error::Error;
use autoagents::core::memory::SlidingWindowMemory;
use autoagents::llm::LLMProvider;
use std::io::{self, Write};
use std::sync::Arc;
use termimad::MadSkin;

pub async fn run_interactive_session(llm: Arc<dyn LLMProvider>) -> Result<(), Error> {
    println!("🚀 Starting Interactive Coding Agent Session");
    println!("💡 Type 'help' for available commands, 'quit' to exit\n");

    // Create memory with larger window for complex tasks
    let memory = Box::new(SlidingWindowMemory::new(30));

    // Create the coding agent
    let coding_agent = CodingAgent {};

    // Build the agent
    let agent = AgentBuilder::new(coding_agent)
        .with_llm(llm)
        .with_memory(memory)
        .build()?;

    // Create environment
    let mut environment = Environment::new(None).await;

    // Register the agent
    let agent_id = environment.register_agent(agent, None).await?;

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("\n🤖 > ");
        let _ = stdout.flush();

        let mut input = String::new();
        if stdin.read_line(&mut input).is_err() {
            println!("\n❌ Failed to read input");
            continue;
        }
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        match input.to_lowercase().as_str() {
            "quit" | "exit" | "q" => {
                println!("👋 Goodbye!");
                break;
            }
            "help" | "h" => {
                print_help();
                continue;
            }
            "clear" => {
                // Clear the terminal
                print!("\x1B[2J\x1B[1;1H");
                continue;
            }
            _ => {
                // Process the task
                println!("\n🔄 Processing your request...\n");

                // Add the task
                let task_id = environment.add_task(agent_id, input).await?;

                // Run the specific task
                match environment.run_task(agent_id, task_id, None).await {
                    Ok(result) => match result.output {
                        Some(output_value) => {
                            match serde_json::from_value::<ReActAgentOutput>(output_value) {
                                Ok(output) => {
                                    let skin = MadSkin::default();
                                    println!("📝 Agent Response:\n");
                                    skin.print_text(&output.response);
                                }
                                Err(err) => println!("❌ Failed to deserialize output: {}", err),
                            }
                        }
                        None => {
                            println!("❌ Task completed but returned no output.");
                        }
                    },
                    Err(e) => {
                        println!("\n❌ Task failed: {}", e);
                    }
                }
            }
        }
    }

    // Shutdown
    environment.shutdown().await;

    Ok(())
}

fn print_help() {
    println!(
        r#"
📚 Interactive Coding Agent Help

This agent uses the ReAct (Reasoning + Acting) pattern to solve coding tasks.

Commands:
  help, h     - Show this help message
  clear       - Clear the terminal
  quit, q     - Exit the session
"#
    );
}
