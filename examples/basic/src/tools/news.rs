#![allow(dead_code, unused_imports)]
use autoagents::{
    llm::{ChatMessage, ChatRole, TextGenerationOptions, LLM},
    providers::ollama::{model::OllamaModel, Ollama},
    tool::{Tool, ToolInputT},
};
use autoagents_derive::{tool, ToolInput};
use futures::stream::StreamExt;
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use urlencoding::encode;

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
                    output
                }
            }
            Err(e) => format!("Failed to parse news results: {}", e),
        }
    } else {
        format!("HTTP request failed with status: {}", response.status())
    }
}
