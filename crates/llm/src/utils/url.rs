use std::fmt::Display;

pub fn create_model_url(base_url: impl Display, api_url: impl Display) -> String {
    format!("{}/{}", base_url, api_url)
}
