use super::field::{Choice, FieldSchemaAttr};
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{parse_macro_input, DeriveInput, LitStr};

pub(crate) struct InputParser {}

impl InputParser {
    pub fn parse(input: TokenStream) -> TokenStream {
        let input = parse_macro_input!(input as DeriveInput);
        let struct_ident = input.ident;
        let mut properties = Vec::new();
        let mut required = Vec::new();

        if let syn::Data::Struct(ref data_struct) = input.data {
            if let syn::Fields::Named(ref fields_named) = data_struct.fields {
                for field in fields_named.named.iter() {
                    let field_name = field.ident.as_ref().unwrap().to_string();
                    // Mark field as required.
                    required.push(format!("\"{}\"", field_name));
                    // Determine JSON schema type from the Rust type.
                    let ty_str = field.ty.clone();
                    let ty_str = ty_str.to_token_stream().clone();
                    let ty_str = ty_str.to_string();
                    let json_type = if ty_str.contains("String") {
                        "string"
                    } else if ty_str.contains("i32")
                        || ty_str.contains("u32")
                        || ty_str.contains("f64")
                    {
                        "number"
                    } else {
                        "string"
                    };
                    // Try to extract field-level metadata.
                    let mut field_desc = None;
                    let mut field_enum = None;
                    for attr in &field.attrs {
                        if attr.path().is_ident("input") {
                            if let Ok(meta) = attr.parse_args::<FieldSchemaAttr>() {
                                field_desc = meta.description;
                                field_enum = meta.choice;
                            } else {
                                panic!("Error with field schema in input");
                            }
                        }
                    }
                    // Build the JSON object for this field.
                    let mut field_schema =
                        format!("\"{}\": {{ \"type\": \"{}\"", field_name, json_type);
                    if let Some(desc) = field_desc {
                        field_schema.push_str(&format!(", \"description\": {}", desc.value()));
                        // Wrap the description in quotes
                        field_schema =
                            field_schema.replace(&desc.value(), &format!("\"{}\"", desc.value()));
                    }
                    if let Some(enum_vals) = field_enum {
                        // We assume the enum value is a valid JSON array string.
                        field_schema.push_str(&format!(
                            ", \"enum\": [{}]",
                            enum_vals
                                .into_iter()
                                .map(|c| match (c, ty_str.as_str()) {
                                    (Choice::String(s), "String") => format!("\"{}\"", s.value()),
                                    (Choice::Number(n), "i32") => format!("\"{}\"", n),
                                    _ => {
                                        panic!("Choices can be only one single type matching the field type")
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join(", ")
                        ));
                    }
                    field_schema.push_str(" }");
                    properties.push(field_schema);
                }
            }
        }
        let properties_str = properties.join(", ");
        let required_str = required.join(", ");
        let schema = format!(
            "{{ \"type\": \"object\", \"properties\": {{ {} }}, \"required\": [{}] }}",
            properties_str, required_str
        );
        let schema_literal = LitStr::new(&schema, struct_ident.span());
        let expanded = quote! {
            impl ToolInputT for #struct_ident {
                fn io_schema() -> &'static str {
                    #schema_literal
                }
            }
        };
        TokenStream::from(expanded)
    }
}
