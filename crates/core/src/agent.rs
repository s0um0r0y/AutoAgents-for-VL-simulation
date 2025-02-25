use async_trait::async_trait;

#[async_trait]
pub trait Agent: Send + Sync {
    fn metadata(&self) -> String;
    fn id(&self) -> String;
    async fn on_message(&self, message: String, ctx: String) -> String;
}
