use async_trait::async_trait;

#[async_trait]
pub trait Transport {
    async fn send(message: String);
}
