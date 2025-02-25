use crate::agent::Agent;
use async_trait::async_trait;

pub struct BasicAgent {
    id: String,
}

impl BasicAgent {
    pub fn new(id: String) -> Self {
        Self { id }
    }
}

#[async_trait]
impl Agent for BasicAgent {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn metadata(&self) -> String {
        todo!()
    }

    async fn on_message(&self, message: String, _ctx: String) -> String {
        println!("On Message in agent {},  {}", self.id(), message);
        "Hello".into()
    }
}
