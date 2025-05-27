# LLM Providers Example

This example demonstrates how to use different LLM providers (Ollama and OpenAI) with the autoagents library.

## Prerequisites

### For Ollama
- Install and run Ollama locally
- Pull a compatible model: `ollama pull qwen2.5:14b`

### For OpenAI
- Set your OpenAI API key as an environment variable:
  ```bash
  export OPENAI_API_KEY=your_api_key_here
  ```

## Usage

Run the example with different providers:

### Using Ollama (local)
```bash
cargo run -- --provider ollama --message "Hello, how are you?"
```

### Using OpenAI
```bash
cargo run -- --provider openai --message "Hello, how are you?"
```

### Comparing both providers
```bash
cargo run -- --provider both --message "Explain quantum computing"
```

### Different modes
Use chat mode (default):
```bash
cargo run -- --provider ollama --mode chat --message "Hello, how are you?"
```

Use text generation mode:
```bash
cargo run -- --provider openai --mode text-gen --message "Once upon a time"
```

### Streaming responses
Add the `--stream` flag to see responses streamed in real-time:

```bash
cargo run -- --provider ollama --message "Tell me a story" --stream
cargo run -- --provider openai --message "Tell me a story" --stream
```

### Custom token limits
Control response length:
```bash
cargo run -- --provider openai --message "Explain AI" --max-tokens 200
```

## Features Demonstrated

1. **Multiple Provider Support**: Shows how to use both local (Ollama) and cloud (OpenAI) providers
2. **Provider Comparison**: Side-by-side comparison of responses from both providers
3. **Chat Completion**: Basic conversation with system and user messages
4. **Text Generation**: Direct text completion without chat context
5. **Streaming**: Real-time response streaming for both providers
6. **Performance Timing**: Response time measurement for each provider
7. **Error Handling**: Proper error handling for API calls
8. **Configuration**: Different models, token limits, and settings for each provider
9. **Common Abstraction**: Both providers use shared OpenAI-style request/response structures for consistency

## Supported Models

### Ollama
- Qwen2.5 14B (default in example)
- DeepSeek R1
- Llama 3.2
- And others (see `OllamaModel` enum)

### OpenAI
- GPT-3.5 Turbo (default in example)
- GPT-4
- GPT-4 Turbo
- GPT-4o
- And others (see `OpenAIModel` enum)

## Customization

You can modify the example to:
- Use different models (change the model in the code)
- Add tool support (register tools before making calls)
- Implement different conversation patterns
- Add custom system prompts
- Compare response quality between providers
- Benchmark performance differences
- Test streaming vs non-streaming modes

## Architecture Notes

This example showcases the unified architecture where both Ollama and OpenAI providers use common OpenAI-style request/response structures from `autoagents_llm::common::openai_types`. This abstraction:

- Eliminates code duplication between providers
- Ensures consistent API across different LLM backends
- Makes it easy to switch between providers or add new OpenAI-compatible ones
- Maintains provider-specific features (e.g., Ollama's `keep_alive`, OpenAI's authentication)