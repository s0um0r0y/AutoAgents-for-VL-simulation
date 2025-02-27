extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{quote, ToTokens};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, DeriveInput, Ident, ItemFn, LitStr, Result, Token, Type,
};

/// This struct holds optional metadata for a field.
struct FieldSchemaAttr {
    description: Option<LitStr>,
    enum_values: Option<LitStr>,
}

impl Parse for FieldSchemaAttr {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut description = None;
        let mut enum_values = None;
        // We'll parse a comma‑separated list of key=value pairs.
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            if key == "description" {
                description = Some(input.parse()?);
            } else if key == "enum_values" {
                enum_values = Some(input.parse()?);
            } else {
                return Err(input.error("unexpected key in tool_io_field attribute"));
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(FieldSchemaAttr {
            description,
            enum_values,
        })
    }
}

#[proc_macro_derive(ToolArg, attributes(arg))]
pub fn arg(input: TokenStream) -> TokenStream {
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
                let ty_str = field.ty.to_token_stream().to_string();
                let json_type = if ty_str.contains("String") {
                    "string"
                } else if ty_str.contains("i32") || ty_str.contains("u32") || ty_str.contains("f64")
                {
                    "number"
                } else {
                    "string"
                };
                // Try to extract field-level metadata.
                let mut field_desc = None;
                let mut field_enum = None;
                for attr in &field.attrs {
                    if attr.path().is_ident("tool_io_field") {
                        if let Ok(meta) = attr.parse_args::<FieldSchemaAttr>() {
                            field_desc = meta.description;
                            field_enum = meta.enum_values;
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
                    field_schema.push_str(&format!(", \"enum\": {}", enum_vals.value()));
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
        impl ToolArg for #struct_ident {
            fn io_schema() -> &'static str {
                #schema_literal
            }
        }
    };
    TokenStream::from(expanded)
}

struct ToolAttributes {
    name: LitStr,
    description: LitStr,
    args: Type,
    output: Type,
}

impl Parse for ToolAttributes {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut name = None;
        let mut description = None;
        let mut args = None;
        let mut output = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "name" => {
                    name = Some(input.parse::<LitStr>()?);
                }
                "description" => {
                    description = Some(input.parse::<LitStr>()?);
                }
                "args" => {
                    args = Some(input.parse::<Type>()?);
                }
                "output" => {
                    output = Some(input.parse::<Type>()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("Unexpected attribute key: {}", other),
                    ))
                }
            }
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }
        Ok(ToolAttributes {
            name: name.ok_or_else(|| syn::Error::new(input.span(), "Missing attribute: name"))?,
            description: description
                .ok_or_else(|| syn::Error::new(input.span(), "Missing attribute: description"))?,
            args: args.ok_or_else(|| syn::Error::new(input.span(), "Missing attribute: args"))?,
            output: output
                .ok_or_else(|| syn::Error::new(input.span(), "Missing attribute: output"))?,
        })
    }
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let tool_attrs = parse_macro_input!(attr as ToolAttributes);
    let input_fn = parse_macro_input!(item as ItemFn);
    let fn_name = &input_fn.sig.ident;
    let tool_name_literal = tool_attrs.name.clone();
    let tool_description = tool_attrs.description;
    let args_type = tool_attrs.args;
    let output_type = tool_attrs.output;
    let tool_struct_ident = Ident::new(&tool_attrs.name.value(), fn_name.span());

    let expanded = quote! {
        #input_fn

        pub struct #tool_struct_ident;

        impl Tool for #tool_struct_ident {
            fn name(&self) -> &'static str {
                #tool_name_literal
            }
            fn description(&self) -> &'static str {
                #tool_description
            }
            fn run(&self, args: Value) -> serde_json::Value {
                let typed_args: #args_type = serde_json::from_value(args)
                    .expect("Failed to deserialize arguments");
                let result: #output_type = #fn_name(typed_args);
                serde_json::to_value(result).expect("Failed to serialize output")
            }
            fn args_schema(&self) -> Value {
                // Retrieve the JSON schema string from the input type.
                    let params_str = <#args_type as ToolArg>::io_schema();
                    // Parse it into a serde_json::Value.
                    let params_value: serde_json::Value = serde_json::from_str(params_str)
                        .expect("Failed to parse parameters schema");
                    params_value
            }
        }

        impl std::fmt::Debug for #tool_struct_ident {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{:?}", self.name())
            }
        }
    };
    expanded.into()
}
