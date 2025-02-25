use crate::llm::LLMConfig;

pub struct LLMContext {
    config: LLMConfig,
}

impl LLMContext {
    pub fn new(config: LLMConfig) -> Self {
        Self { config }
    }
}
