pub struct LLMContext {
    config: String,
}

impl LLMContext {
    pub fn new(config: String) -> Self {
        Self { config }
    }
}
