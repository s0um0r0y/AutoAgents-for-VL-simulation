use crate::agent::{Agent, AgentDeriveT, AgentT};
use async_trait::async_trait;
use autoagents_llm::llm::{ChatMessage, ChatRole, LLM};

const REACT_PROMPT: &str = r#"
You are a highly capable AI assistant that follows a ReAct (Reasoning and Acting) approach to solving problems.
Your process is as follows:
1. Carefully analyze the user's input and think step-by-step about the problem.
2. If you need to perform any specific actions use tools provided to you.
3. After receiving the results from your action, integrate the observations into your ongoing reasoning.
4. Continue this iterative process until you are ready to provide a comprehensive final answer.

IMPORTANT
When you are ready to deliver your final answer, begin your response with "FINAL:
"#;

#[async_trait]
pub trait ReActAgentT: AgentT {
    async fn step<L: LLM + 'static>(
        &self,
        llm: &mut L,
        messages: Vec<ChatMessage>,
    ) -> Result<String, Self::Err>;
}

impl<T, L> Agent<'_, T, L>
where
    T: AgentDeriveT + AgentT + ReActAgentT,
    L: LLM + 'static,
{
    pub async fn run_react(&mut self, prompt: &str) -> Result<T::Output, T::Err> {
        let mut messages = vec![
            ChatMessage {
                role: ChatRole::System,
                content: REACT_PROMPT.into(),
            },
            // ChatMessage {
            //     role: ChatRole::System,
            //     content: self.description().to_string(),
            // },
            ChatMessage {
                role: ChatRole::User,
                content: prompt.into(),
            },
        ];

        loop {
            println!("In Loop Calling: {:?}", messages);
            let response = self
                .inner
                .chat_completion(self.llm, messages.clone())
                .await
                .unwrap();
            let response_string = response.clone().message.content;
            println!("REsponse in loop {:?}", response);

            // If the response starts with "FINAL:" or there's no action, consider it the final answer.
            if response_string.starts_with("FINAL:") {
                println!("Final Answer: {}", response);
                todo!()
            }

            match response.message.role {
                ChatRole::Assistant => {
                    let tools_call = response.message.tool_calls;
                    for tool in tools_call {
                        let tool_func = &tool.function;
                        let tool_result = self
                            .llm
                            .call_tool(&tool_func.name, tool_func.arguments.clone());
                        // Append the observation from the tool call to the conversation history.
                        messages.push(ChatMessage {
                            role: ChatRole::Tool,
                            content: format!(
                                "Observation From Tool {} \n: {}",
                                tool.function.name,
                                tool_result.unwrap()
                            ),
                        });
                    }
                }
                _ => {
                    //Ignore
                }
            }
        }
    }
}
