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
}
