use serde_json::Value;
use std::fmt::Debug;

pub trait Tool: Send + Sync + Debug {
    /// The name of the tool.
    fn name(&self) -> &'static str;
    /// A description explaining the tool’s purpose.
    fn description(&self) -> &'static str;
    /// Return a description of the expected arguments.
    fn args_schema(&self) -> Value;
    /// Run the tool with the given arguments (in JSON) and return the result (in JSON).
    fn run(&self, args: Value) -> Value;
}

pub trait ToolArg {
    fn io_schema() -> &'static str;
}
