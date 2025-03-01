use std::fmt::Display;

pub fn create_model_url(base_url: impl Display, api_url: impl Display) -> String {
    format!("{}/{}", base_url, api_url)
}

#[test]
fn test_create_model_url() {
    assert_eq!(
        "test_base/api_base",
        create_model_url("test_base", "api_base")
    )
}
