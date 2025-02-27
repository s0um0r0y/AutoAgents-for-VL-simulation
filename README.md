# AutoAgents

**AutoAgents** is a distributed multi-agent framework written in Rust that enables you to build, deploy, and coordinate multiple intelligent agents. AutoAgents leverages modern language models and tool integrations to provide robust, scalable, and flexible solutions for various use cases—from natural language processing and automated chat to advanced tool orchestration.

---

## Features

- **Distributed Architecture:** Coordinate multiple agents across different nodes.
- **Multi-Agent Coordination:** Allow agents to interact, share knowledge, and collaborate on complex tasks.
- **Tool Integration:** Seamlessly integrate with external tools and services to extend agent capabilities.
- **Extensible Framework:** Easily add new providers and tools using our intuitive plugin system.

---

## Supported Providers

AutoAgents currently supports the **Ollama** provider. The following table summarizes the supported models:

| Provider | Supported Models               |
|----------|--------------------------------|
| Ollama   | Llama-3.2, DeepSeek-R1-8B      |

*Note: Provider support is actively evolving.*


```sh
git clone https://github.com/yourusername/autoagents.git
cd autoagents
cargo build --release
```
