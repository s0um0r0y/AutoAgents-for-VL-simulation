use actix_web::{
    error::ErrorInternalServerError,
    web::{self, Bytes},
    App, HttpResponse, HttpServer, Responder,
};
use autoagents::{
    llm::{ChatMessage, ChatRole, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::ToolArg,
    Tool,
};
use autoagents_derive::{tool, ToolArg};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: Option<bool>,
}

#[derive(Serialize)]
struct ChatChunk {
    id: String,
    object: String,
    created: u64,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    delta: Delta,
    index: u8,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    pub role: ChatRole,
    pub content: String,
}

#[derive(Serialize, Deserialize, ToolArg)]
pub struct GetCurrentWeatherArgs {}

#[tool(
    name = "GetCurrentWeather",
    description = "Use this tool to get the current Weather",
    args = GetCurrentWeatherArgs,
    output = String
)]
fn get_current_weather(_args: GetCurrentWeatherArgs) -> String {
    // Dummy implementation.
    format!("Current Time is {:?}", SystemTime::now())
}

async fn chat_completion_endpoint(req: web::Json<ChatRequest>) -> impl Responder {
    let request = req.into_inner();
    let stream_enabled = request.stream.unwrap_or(false);

    // Initialize the LLM with the Llama3.2 model and register the tool.
    let mut llm = Ollama::new().with_model(OllamaModel::Llama3_2);
    llm.register_tool(GetCurrentWeather);

    let messages = request.messages.clone();

    // Helper closure to generate a timestamp and ID.
    let generate_timestamp_id = || -> (u64, String) {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        (start_time, format!("chatcmpl-{}", start_time))
    };

    if stream_enabled {
        // For SSE, create a stream that maps each LLM chunk to an SSE message.
        let sse_stream =
            llm.chat_completion_stream(messages, None)
                .await
                .map(move |chunk_result| match chunk_result {
                    Ok(resp) => {
                        let (start_time, id) = generate_timestamp_id();

                        //If the message role is assistant, check for tool calls.
                        if let Some(tool_call) = resp.message.tool_calls.first() {
                            if let Some(tool_result) = llm.call_tool(
                                &tool_call.function.name,
                                tool_call.function.arguments.clone(),
                            ) {
                                // If a tool call produced a result, build a chunk with that output.
                                let tool_chunk = ChatChunk {
                                    id,
                                    object: "chat.completion.chunk".to_string(),
                                    created: start_time,
                                    model: resp.model,
                                    choices: vec![Choice {
                                        delta: Delta {
                                            role: ChatRole::Assistant,
                                            content: tool_result.to_string(),
                                        },
                                        index: 0,
                                        finish_reason: Some("stop".to_string()),
                                    }],
                                };
                                let json = serde_json::to_string(&tool_chunk)
                                    .unwrap_or_else(|_| "{}".into());
                                return Ok(Bytes::from(format!("data: {}\n\n", json)));
                            }
                        }

                        // Otherwise, return the base chunk.
                        // Build the base chunk using the LLM's response.
                        let base_chunk = ChatChunk {
                            id: id.clone(),
                            object: "chat.completion.chunk".to_string(),
                            created: start_time,
                            model: resp.model.clone(),
                            choices: vec![Choice {
                                delta: Delta {
                                    role: ChatRole::Assistant,
                                    content: resp.message.content.clone(),
                                },
                                index: 0,
                                finish_reason: if resp.done {
                                    Some("stop".to_string())
                                } else {
                                    None
                                },
                            }],
                        };
                        let json =
                            serde_json::to_string(&base_chunk).unwrap_or_else(|_| "{}".into());
                        Ok(Bytes::from(format!("data: {}\n\n", json)))
                    }
                    Err(e) => Err(ErrorInternalServerError(e)),
                });

        HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(sse_stream)
    } else {
        // Handle non-streaming response.
        match llm.chat_completion(messages, None).await {
            Ok(resp) => {
                let (start_time, id) = generate_timestamp_id();
                let response_body = json!({
                    "id": id,
                    "object": "chat.completion",
                    "created": start_time,
                    "model": request.model,
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": resp.message.content
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }]
                });
                HttpResponse::Ok()
                    .content_type("application/json")
                    .json(response_body)
            }
            Err(e) => HttpResponse::InternalServerError().body(format!("Error: {}", e)),
        }
    }
}

async fn fetch_models() -> impl Responder {
    // Hard-coded list of models for demonstration purposes.
    let models = json!({
        "data": [
            {
                "id": "autoagent",
                "object": "model",
                "owned_by": "liquidos",
                "permission": []
            }
        ]
    });
    HttpResponse::Ok().json(models)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("Server running at http://127.0.0.1:8080");
    HttpServer::new(|| {
        App::new()
            .route(
                "/v1/chat/completions",
                web::post().to(chat_completion_endpoint),
            )
            .route("/v1/models", web::get().to(fetch_models))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
