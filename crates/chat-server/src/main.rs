use actix_web::{
    error::ErrorInternalServerError,
    web::{self, Bytes},
    App, HttpResponse, HttpServer, Responder,
};
use autoagents::{
    llm::{ChatMessage, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
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
    content: String,
}

async fn chat_completion_endpoint(req: web::Json<ChatRequest>) -> impl Responder {
    let request = req.into_inner();
    let stream_enabled = request.stream.unwrap_or(false);
    let llm = Ollama::new().with_model(OllamaModel::Llama3_2);
    let messages = request.messages.clone();

    if stream_enabled {
        let sse_stream = llm
            .chat_completion_stream(messages, None)
            .await
            .map(|chunk_result| match chunk_result {
                Ok(resp) => {
                    let start_time = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let chunk = ChatChunk {
                        id: format!("chatcmpl-{}", start_time),
                        object: "chat.completion.chunk".to_string(),
                        created: start_time,
                        model: resp.model.clone(),
                        choices: vec![Choice {
                            delta: Delta {
                                content: resp.message.content,
                            },
                            index: 0,
                            finish_reason: if resp.done {
                                Some("stop".to_string())
                            } else {
                                None
                            },
                        }],
                    };
                    let json = serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".into());
                    Ok(Bytes::from(format!("data: {}\n\n", json)))
                }
                Err(e) => Err(ErrorInternalServerError(e)),
            });

        HttpResponse::Ok()
            .content_type("text/event-stream")
            .streaming(sse_stream)
    } else {
        // Non-streaming response
        let resp = llm.chat_completion(messages, None).await.unwrap();
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let response = json!({
            "id": format!("chatcmpl-{}", start_time),
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
            .json(response)
    }
}

/// Fetch models endpoint mimicking OpenAI's GET /v1/models API.
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
