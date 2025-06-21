# Basic Example

This example demonstrates various AutoAgents features including chat completion, tool usage, text generation, and different agent types.

## Prerequisites

1. Install and run Ollama locally
2. Pull a compatible model: `ollama pull qwen2.5:14b`
3. For the news search tool, set the `NEWS_API_KEY` environment variable

## Use Cases

### Chat Completion (Stream)
```sh
cargo run --package basic -- --usecase chat
```
Demonstrates streaming chat completion with tool support.

### Tools
```sh
cargo run --package basic -- --usecase tool
```
Shows how to register and use tools with LLMs.

### Text Generation
```sh
cargo run --package basic -- --usecase text-gen
```
Demonstrates text generation with streaming support.

### Basic Agent
```sh
cargo run --package basic -- --usecase agent
```
Shows a basic event-driven agent that can search for news articles.

### ReAct Agent
```sh
cargo run --package basic -- --usecase react
```
Demonstrates a ReAct (Reasoning and Acting) agent that solves problems step-by-step with reasoning traces.

### Chain of Thought (CoT) Agent
```sh
cargo run --package basic -- --usecase cot
```
