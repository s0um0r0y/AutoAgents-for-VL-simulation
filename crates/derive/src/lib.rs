extern crate proc_macro;
use proc_macro::TokenStream;
use tool::{input::InputParser, ToolParser};
mod tool;

#[proc_macro_derive(ToolInput, attributes(input))]
pub fn input(input: TokenStream) -> TokenStream {
    InputParser::default().parse(input)
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    ToolParser::default().parse(attr, item)
}
