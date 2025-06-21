#![allow(dead_code)]
use actix_web::{
    error::ErrorInternalServerError,
    web::{self, Bytes},
    App, HttpResponse, HttpServer, Responder,
};
use autoagents::{
    llm::{ChatMessage, ChatRole, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{tool, ToolInput};
use futures::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value;
use std::time::{SystemTime, UNIX_EPOCH};
use urlencoding::encode;

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

#[derive(Serialize, Deserialize, ToolInput)]
pub struct GetCurrentWeatherArgs {}

#[tool(
    name = "GetCurrentWeather",
    description = "Use this tool to get the current Weather",
    input = GetCurrentWeatherArgs,
    output = String
)]
fn get_current_weather(_args: GetCurrentWeatherArgs) -> String {
    // Dummy implementation.
    format!("Current Time is {:?}", SystemTime::now())
}

#[derive(Serialize, Deserialize, ToolInput)]
pub struct SearchNewsArgs {
    #[input(description = "Query to use for the news Search")]
    pub query: String,
}

#[derive(Deserialize)]
struct NewsArticleSource {
    id: Option<String>,
    name: Option<String>,
}

#[derive(Deserialize)]
struct NewsArticle {
    source: NewsArticleSource,
    author: Option<String>,
    title: String,
    description: Option<String>,
    url: String,
    #[serde(rename = "urlToImage")]
    url_to_image: Option<String>,
    #[serde(rename = "publishedAt")]
    published_at: String,
    content: String,
}

#[derive(Deserialize)]
struct NewsApiResponse {
    articles: Vec<NewsArticle>,
}

#[tool(
    name = "SearchNews",
    description = "Use this tool to search News using for the given query",
    input = SearchNewsArgs,
    output = String
)]
fn search_news(args: SearchNewsArgs) -> String {
    println!("Search Tool Query {}", args.query);
    // Retrieve your News API key from the environment.
    let api_key = std::env::var("NEWS_API_KEY").expect("NEWS_API_KEY environment variable not set");

    // Encode the query to be URL-safe.
    let query_encoded = encode(&args.query);

    // Build the URL using the News API endpoint.
    let url = format!(
        "https://newsapi.org/v2/everything?q={}&sortBy=publishedAt&apiKey={}&pageSize=5",
        query_encoded, api_key
    );

    // Perform the GET request using ureq.
    let mut response = ureq::get(&url).call().unwrap();
    if response.status() == StatusCode::OK {
        // Deserialize the JSON response into our response struct.
        // println!("Respnse {:?}", response.body_mut().read_to_string());
        let news_response: Result<NewsApiResponse, _> = response.body_mut().read_json();
        match news_response {
            Ok(resp) => {
                if resp.articles.is_empty() {
                    "No news articles found.".to_string()
                } else {
                    let mut output = String::new();
                    for article in resp.articles {
                        output.push_str(&format!(
                            "Title: {}\nDescription: {}\nURL: {}\n\n Content: {}",
                            article.title,
                            article
                                .description
                                .unwrap_or_else(|| "No description".to_string()),
                            article.url,
                            article.content
                        ));
                    }
                    let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_32B);
                    let resp = llm.chat_completion_sync(
                        vec![
                            ChatMessage {
                                role: ChatRole::System,
                                content:
                                    "You are an Assistant who can summarize given information into a markdown format"
                                        .into(),
                            },
                            ChatMessage {
                                role: ChatRole::User,
                                content: format!("Summarize the below google search data for the query '{}': \n {}", args.query, output),
                            },
                        ],
                        None,
                    );
                    let summarized_val = format!("{}", resp.unwrap().message.content);
                    summarized_val
                }
            }
            Err(e) => format!("Failed to parse news results: {}", e.to_string()),
        }
    } else {
        format!("HTTP request failed with status: {}", response.status())
    }
}

#[derive(Serialize, Deserialize, ToolInput)]
pub struct SearchGoogleArgs {
    #[input(description = "Query to use for the google Search")]
    pub query: String,
}

#[derive(Deserialize)]
struct GoogleSearchResponse {
    items: Option<Vec<GoogleSearchItem>>,
}

#[derive(Deserialize)]
struct GoogleSearchItem {
    title: String,
    snippet: String,
    link: String,
}

#[tool(
    name = "SearchGoogle",
    description = "Use this tool to search the internet with user queries",
    input = SearchGoogleArgs,
    output = String
)]
fn search_google(args: SearchGoogleArgs) -> String {
    // Retrieve the API key and Custom Search Engine ID (CX) from environment variables.
    let api_key =
        std::env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set");
    let cx = std::env::var("GOOGLE_CX").expect("GOOGLE_CX environment variable not set");

    // URL-encode the query.
    println!("Tool Query {}", args.query);
    let query_encoded = encode(&args.query);

    // Build the URL using the Custom Search API endpoint.
    let url = format!(
        "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}",
        api_key, cx, query_encoded
    );

    // Perform the GET request synchronously using ureq.
    let mut response = ureq::get(&url).call().unwrap();

    if response.status() == StatusCode::OK {
        // Deserialize the JSON response.
        let search_response: Result<GoogleSearchResponse, _> = response.body_mut().read_json();
        match search_response {
            Ok(result) => {
                if let Some(items) = result.items {
                    let mut output = String::new();
                    for item in items {
                        output.push_str(&format!(
                            "Title: {}\nSnippet: {}\nLink: {}\n\n",
                            item.title, item.snippet, item.link
                        ));
                    }
                    let llm = Ollama::new().set_model(OllamaModel::Qwen2_5_32B);
                    let resp = llm.chat_completion_sync(
                        vec![
                            ChatMessage {
                                role: ChatRole::System,
                                content:
                                    "You are an Assistant who can summarize given information into a markdown format"
                                        .into(),
                            },
                            ChatMessage {
                                role: ChatRole::User,
                                content: format!("Summarize the below google search data for the query '{}': \n {}", args.query, output),
                            },
                        ],
                        None,
                    );

                    let summarized_val = format!("{}", resp.unwrap().message.content);
                    // println!("DATA {}", summarized_val.clone());
                    summarized_val
                } else {
                    "No results found.".to_string()
                }
            }
            Err(e) => format!("Failed to parse search results: {}", e),
        }
    } else {
        format!("HTTP request failed with status: {}", response.status())
    }
}

async fn chat_completion_endpoint(req: web::Json<ChatRequest>) -> impl Responder {
    let request = req.into_inner();
    let stream_enabled = request.stream.unwrap_or(false);

    // Initialize the LLM with the Llama3.2 model and register the tool.
    let mut llm = Ollama::new().set_model(OllamaModel::Qwen2_5_32B);
    llm.register_tool(Box::new(GetCurrentWeather));
    llm.register_tool(Box::new(SearchGoogle));
    llm.register_tool(Box::new(SearchNews));

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
                                let content = match tool_result.as_str() {
                                    Some(s) => s.to_owned(),
                                    None => tool_result.to_string(),
                                };
                                let tool_chunk = ChatChunk {
                                    id,
                                    object: "chat.completion.chunk".to_string(),
                                    created: start_time,
                                    model: resp.model,
                                    choices: vec![Choice {
                                        delta: Delta {
                                            role: ChatRole::Assistant,
                                            content: content,
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
