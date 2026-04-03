# A2A Plan/Todo Multi-Agent Example

This example demonstrates an **Agent-to-Agent (A2A)** multi-agent system with
a planning/todo workflow built with `supervisors.rs`, using the SDK properly:

- **`LoopAgent`** for ReAct-style reasoning loops
- **`FunctionCallingExtension`** for tool registration and invocation
- **`Agent.send()`** for A2A communication through the shared `Supervisor`

| Agent | Role | Tools |
|-------|------|-------|
| **planner_agent** | Main orchestrator — plans tasks via `todo` tool | `todo`, `send_task` (A2A) |
| **write_file_agent** | File writing specialist | `write_file` |
| **read_file_agent** | File reading specialist | `read_file` |
| **shell_agent** | Directory specialist | `pwd`, `ls` |

## Quick Start

```bash
cd examples/a2a_plan_todo

# 1. Create your .env with an OpenAI API key
cp .env.example .env
# Edit .env and fill in your OPENAI_API_KEY

# 2. Run with uv (installs deps + builds supervisor automatically)
uv run python main.py
```

## How It Works

1. The user types a message in the terminal.
2. **planner_agent** (a `LoopAgent`) receives it via `handle_message()`, which triggers `run_loop()`.
3. Each `step()` calls the LLM. If tool_calls are returned, they are executed via `FunctionCallingExtension.call_tool()`.
4. If the planner decides to delegate, it calls `send_task` which uses `Agent.send()` to route a `Message` through the shared `Supervisor`.
5. `supervisor.run_once()` dispatches the message to the target sub-agent.
6. The sub-agent's `handle_message()` runs its own ReAct loop and sends the result back via `self.send()`.
7. **planner_agent** synthesises the final answer.

## Architecture

```
User ──→ planner_agent (LoopAgent + FunctionCallingExtension)
              ├── todo (local tool via FunctionCallingExtension)
              └── send_task (A2A via Agent.send → Supervisor → sub-agent)
                      ├──→ write_file_agent (LoopAgent + write_file tool)
                      ├──→ read_file_agent  (LoopAgent + read_file tool)
                      └──→ shell_agent      (LoopAgent + pwd/ls tools)
```

All agents are registered with a single `Supervisor` instance backed by a
tokio async runtime in Rust, providing fault-tolerant message routing.

## SDK Features Used

| Feature | Usage |
|---------|-------|
| `LoopAgent` | Base class for all 4 agents — provides `step()`, `on_loop_start/end`, `run_loop()` |
| `FunctionCallingExtension` | Tool registration (`register_tool`) and invocation (`call_tool`) |
| `Agent.send()` | A2A communication — wraps `Message` and routes via `Supervisor.send()` |
| `Supervisor` | Message routing, `run_once()` for dispatch |
| `Message` | Inter-agent communication payload |

## Commands

- Type a message to interact with the planner agent
- `todos` — show the current todo list
- `quit` / `exit` / `q` — exit the program
