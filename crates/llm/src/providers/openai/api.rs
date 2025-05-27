use strum::{Display, EnumString};

#[derive(Debug, EnumString, Display, Clone)]
pub enum OpenAIAPI {
    #[strum(serialize = "v1/completions")]
    TextGenerate,
    #[strum(serialize = "v1/chat/completions")]
    ChatCompletion,
}