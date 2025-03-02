use std::fmt::Debug;
use strum::{Display, EnumString};

#[derive(Debug, EnumString, Display, Clone)]
pub enum OllamaModel {
    #[strum(serialize = "deepseek-r1:8b")]
    DeepSeekR18B,
    #[strum(serialize = "deepseek-r1")]
    DeepSeekR1,
    #[strum(serialize = "llama3.2")]
    Llama3_2,
    #[strum(serialize = "qwen2.5:32b")]
    Qwen2_5_32B,
}

impl OllamaModel {
    pub fn supports_tools(&self) -> bool {
        matches!(&self, OllamaModel::Qwen2_5_32B | OllamaModel::Llama3_2)
    }
}
