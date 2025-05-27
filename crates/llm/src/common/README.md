# Common OpenAI-Style Types

This module provides shared OpenAI-style request/response structures that are used by multiple LLM providers in the autoagents library.

## Overview

Both Ollama and OpenAI providers use OpenAI-compatible API formats, so this module abstracts the common structures to avoid code duplication and ensure consistency across providers.

## Key Components

### Request Types

- `OpenAIStyleChatCompletionRequest` - Chat completion request structure
- `OpenAIStyleMessage` - Individual chat message
- `OpenAIStyleTool` / `OpenAIStyleFunction` - Tool/function calling definitions

### Response Types

- `OpenAIStyleChatCompletionResponse` - Chat completion response
- `OpenAIStyleStreamResponse` - Streaming response structure
- `OpenAIStyleChatChoice` / `OpenAIStyleStreamChoice` - Response choices

### Options

- `OpenAIStyleChatCompletionOptions` - Chat completion parameters
- `OpenAIStyleTextGenerationOptions` - Text generation parameters

## Provider Usage

### Ollama Provider
Uses the common types directly since Ollama supports OpenAI-compatible endpoints:
```rust
let request = OpenAIStyleChatCompletionRequest::from_chat_messages(messages, model)
    .set_chat_options(options)
    .set_keep_alive("5m".to_string()); // Ollama-specific
```

### OpenAI Provider
Uses the common types with OpenAI-specific authentication:
```rust
let request = OpenAIStyleChatCompletionRequest::from_chat_messages(messages, model)
    .set_chat_options(options);
// Headers: Authorization: Bearer <api_key>
```

## Benefits

1. **Code Reuse** - Eliminates duplication between providers
2. **Consistency** - Ensures same request/response format across providers
3. **Maintainability** - Single source of truth for OpenAI-style structures
4. **Extensibility** - Easy to add new providers that use OpenAI-compatible APIs

## Stream Parsing

The module includes `parse_openai_style_stream_chunk()` to handle Server-Sent Events format used by both OpenAI and Ollama for streaming responses.

## Provider-Specific Adaptations

While using common structures, each provider can add its own specific fields:

- **Ollama**: `keep_alive`, `format` fields
- **OpenAI**: Authentication headers, specific model parameters

This design allows maximum code reuse while preserving provider-specific functionality.