use reqwest::Error;
use serde_json::Value;

pub(crate) struct HTTPRequest {}

impl HTTPRequest {
    pub async fn request(url: String, body: Value) -> Result<String, Error> {
        let client = reqwest::Client::new();
        let response = client.post(&url).json(&body).send().await?;
        response.text().await
    }
}
