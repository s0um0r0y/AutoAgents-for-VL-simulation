use std::fmt::Debug;
use strum::{Display, EnumString};

#[derive(Debug, EnumString, Display, Clone)]
pub enum OpenAIModel {
    #[strum(serialize = "gpt-4")]
    GPT4,
    #[strum(serialize = "gpt-4-turbo")]
    GPT4Turbo,
    #[strum(serialize = "gpt-4o")]
    GPT4O,
    #[strum(serialize = "gpt-4o-mini")]
    GPT4OMini,
    #[strum(serialize = "gpt-3.5-turbo")]
    GPT35Turbo,
    #[strum(serialize = "text-davinci-003")]
    TextDavinci003,
    #[strum(serialize = "text-davinci-002")]
    TextDavinci002,
}

impl OpenAIModel {
    pub fn supports_tools(&self) -> bool {
        matches!(
            &self,
            OpenAIModel::GPT4
                | OpenAIModel::GPT4Turbo
                | OpenAIModel::GPT4O
                | OpenAIModel::GPT4OMini
                | OpenAIModel::GPT35Turbo
        )
    }

    pub fn supports_chat(&self) -> bool {
        matches!(
            &self,
            OpenAIModel::GPT4
                | OpenAIModel::GPT4Turbo
                | OpenAIModel::GPT4O
                | OpenAIModel::GPT4OMini
                | OpenAIModel::GPT35Turbo
        )
    }
}