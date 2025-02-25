use strum::{Display, EnumString};

#[derive(Debug, EnumString, Display, Clone)]
pub enum OllamaModel {
    #[strum(serialize = "deepseek-r1:8b")]
    DeepSeekR18B,
}
