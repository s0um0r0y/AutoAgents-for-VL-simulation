use strum::{Display, EnumString};

#[derive(Debug, EnumString, Display, Clone)]
pub enum OllamaAPI {
    #[strum(serialize = "api/generate")]
    TextGenerate,
    #[strum(serialize = "api/chat")]
    ChatCompletion,
}
