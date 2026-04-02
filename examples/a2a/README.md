# A2A Multi-Agent Example

This example demonstrates an **Agent-to-Agent (A2A)** multi-agent system built
with `supervisor.rs`.  Three agents collaborate via ReAct-style reasoning
powered by OpenAI:

| Agent | Role | Tools |
|-------|------|-------|
| **main_agent** | Orchestrator – routes user queries | `bash`, delegates to sub-agents |
| **weather_agent** | Weather specialist | `get_weather` (via wttr.in) |
| **search_agent** | Web search specialist | `web_search` (via DuckDuckGo) |

Sub-agents can also delegate to each other (e.g. weather\_agent can ask
search\_agent for supplementary info and vice-versa).

## Quick Start

```bash
cd examples/a2a

# 1. Create your .env with an OpenAI API key
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY

# 2. Run with uv (installs deps + builds supervisor automatically)
uv run python main.py
```

## How It Works

1. The user types a message in the terminal.
2. **main_agent** receives it and enters a ReAct loop (Thought → Action → Observation).
3. If the query involves weather, main\_agent delegates to **weather\_agent**.
4. If the query involves web search, main\_agent delegates to **search\_agent**.
5. Sub-agents run their own ReAct loops and return results.
6. Sub-agents can also cross-delegate to each other.
7. **main\_agent** synthesises the final answer and prints it.

## Architecture

```
User ←→ main_agent (bash tool)
              ├──→ weather_agent (get_weather tool)
              │         └──→ search_agent (cross-delegation)
              └──→ search_agent (web_search tool)
                        └──→ weather_agent (cross-delegation)
```

All agents are registered with a single `Supervisor` instance backed by a
tokio async runtime in Rust, providing fault-tolerant message routing.
