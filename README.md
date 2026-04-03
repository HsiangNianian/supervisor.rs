<div align="center">

# supervisors.rs

**A composable multi-agent framework powered by Rust and Python.**

Build, orchestrate, and scale intelligent agents with a high-performance
Rust core and a flexible Python API.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-stable-orange.svg)](https://www.rust-lang.org/)
[![Docs](https://readthedocs.org/projects/supervisors-rs/badge/?version=latest)](https://supervisor-rs.readthedocs.io/)

[Getting Started](#getting-started) |
[Agent Types](#agent-types) |
[Examples](#examples) |
[Extensions](#extensions) |
[API Reference](#api-reference) |
[Contributing](#contributing)

</div>

---

## Overview

**supervisors.rs** is a cross-language agent framework designed for building
production-grade AI agent systems.  The core is written in Rust for
performance and safety, while the Python API provides ergonomic abstractions
for rapid development.

The framework supports four fundamental agent patterns that can be used
independently or composed together to solve complex, real-world problems:

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Loop Agent** | LLM + iterative reasoning loop | ReAct agents, chain-of-thought, self-refinement |
| **Pipeline Agent** | Sequential stage-based processing | ETL, data transformation, multi-step workflows |
| **Supervisors + SubAgents** | Hierarchical delegation | Task routing, modular decomposition, microservices |
| **Multi-Agent** | Collaborative peer agents | Research teams, debate, consensus-building |

All four patterns are composable -- a `SupervisorsAgent` can manage
`MultiAgent` teams as sub-agents, a `LoopAgent` can embed pipeline logic,
and any agent type can be registered on the same `Supervisors` instance.

---

## Getting Started

### Installation

Build and install with [maturin](https://github.com/PyO3/maturin):

```bash
pip install maturin
maturin develop          # editable install for development
```

### Quick Start

```python
from supervisors import Agent, Supervisor, Message

class GreeterAgent(Agent):
    def handle_message(self, msg: Message) -> None:
        print(f"Hello from '{self.name}'! You said: {msg.content}")

sup = Supervisor()
greeter = GreeterAgent("greeter")
greeter.register(sup)

sup.send(Message("user", "greeter", "Hi there!"))
sup.run_once()
# Output: Hello from 'greeter'! You said: Hi there!
```

---

## Agent Types

### Loop Agent

The `LoopAgent` runs an iterative reasoning loop over each incoming message.
Override the `step()` method to define per-iteration logic.

```python
from supervisors import LoopAgent, Message

class ReasoningAgent(LoopAgent):
    def step(self, state):
        state["count"] = state.get("count", 0) + 1
        if state["count"] >= 3:
            state["done"] = True
            state["result"] = "Reasoning complete."
        return state

agent = ReasoningAgent("reasoner", max_iterations=10)
state = agent.run_loop(Message("user", "reasoner", "Think about X"))
print(state["result"])
```

### Pipeline Agent

The `PipelineAgent` processes messages through an ordered sequence of
stage functions.

```python
from supervisors import PipelineAgent, Message

def parse(ctx):
    ctx["tokens"] = ctx["input"].split()
    return ctx

def analyse(ctx):
    ctx["count"] = len(ctx["tokens"])
    return ctx

agent = PipelineAgent("analyser", stages=[parse, analyse])
result = agent.run_pipeline(Message("user", "analyser", "hello world"))
print(result["count"])  # 2
```

### Supervisor + SubAgents

The `SupervisorAgent` routes incoming messages to specialised sub-agents
based on a routing function.

```python
from supervisors import Agent, SupervisorAgent, Message, Supervisor

class Worker(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] handling: {msg.content}")

sup = Supervisor()
manager = SupervisorAgent(
    "manager",
    router=lambda msg: "worker_a" if "urgent" in msg.content else "worker_b",
)
manager.add_sub_agent(Worker("worker_a"))
manager.add_sub_agent(Worker("worker_b"))
manager.register(sup)

sup.send(Message("user", "manager", "urgent: fix production"))
sup.run_once()
# Output: [worker_a] handling: urgent: fix production
```

### Multi-Agent

The `MultiAgent` coordinates a group of peer agents that collaborate
to solve problems.

```python
from supervisors import Agent, MultiAgent, Message, Supervisor

class Researcher(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] researching: {msg.content}")

sup = Supervisor()
team = MultiAgent(
    "research_team",
    members=[Researcher("alice"), Researcher("bob")],
    max_rounds=5,
)
team.register(sup)

sup.send(Message("user", "research_team", "Investigate topic X"))
sup.run_once()
```

### Composing Agents

The real power emerges when you combine patterns.  For example, a
`SupervisorAgent` can manage `MultiAgent` teams as sub-agents:

```python
from supervisors import (
    Agent, MultiAgent, SupervisorAgent, Message, Supervisor,
)

class Specialist(Agent):
    def handle_message(self, msg):
        print(f"[{self.name}] working on: {msg.content}")

# Build a collaborative team.
team = MultiAgent("dev_team", members=[
    Specialist("frontend"),
    Specialist("backend"),
])

# Wrap it in a hierarchical supervisor.
manager = SupervisorAgent("manager", router=lambda msg: "dev_team")
manager.add_sub_agent(team)

sup = Supervisor()
manager.register(sup)
sup.send(Message("cto", "manager", "Build feature X"))
sup.run_once()
```

---

## Examples

Each example is a self-contained project in the `examples/` directory:

| Example | Pattern | Description |
|---------|---------|-------------|
| [`loop/`](examples/loop/) | Loop Agent | Customer support with iterative reasoning |
| [`pipeline/`](examples/pipeline/) | Pipeline Agent | Log processing with sequential stages |
| [`supervisor_subagent/`](examples/supervisor_subagent/) | Supervisor + SubAgents | Content moderation with routing |
| [`multi_agent/`](examples/multi_agent/) | Multi-Agent | Collaborative research team |
| [`composite/`](examples/composite/) | Supervisor + Multi-Agent | DevOps incident response system |
| [`a2a/`](examples/a2a/) | A2A + ReAct | Multi-agent chat with OpenAI (requires API key) |

Run any example:

```bash
cd examples/loop
python main.py
```

---

## Extensions

Extensions add capabilities to any agent via the plugin system.  Load
extensions with `agent.use(extension)`:

### RAG (Retrieval-Augmented Generation)

```python
from supervisors.ext.rag import RAGExtension

class MyRAG(RAGExtension):
    def __init__(self):
        super().__init__(auto_retrieve=True, top_k=5)
        self._store = []

    def retrieve(self, query, top_k=None):
        return [d for d in self._store if query.lower() in d.lower()]

    def add_documents(self, docs, **kwargs):
        self._store.extend(docs)

agent.use(MyRAG())
```

### Function Calling

```python
from supervisors.ext.function_calling import FunctionCallingExtension

fc = FunctionCallingExtension()

@fc.tool(description="Add two numbers")
def add(a: int, b: int) -> int:
    return a + b

agent.use(fc)
result = fc.call_tool("add", a=1, b=2)  # returns 3
```

### MCP (Model Context Protocol)

```python
from supervisors.ext.mcp import MCPExtension

mcp = MCPExtension(server_url="http://localhost:8080")

@mcp.mcp_tool(description="Reverse a string")
def reverse(text: str) -> str:
    return text[::-1]

agent.use(mcp)
```

### Skills

```python
from supervisors.ext.skills import SkillsExtension

skills = SkillsExtension()

@skills.skill
def summarise(agent, msg):
    return f"Summary: {msg.content[:50]}..."

agent.use(skills)
```

### A2A (Agent-to-Agent)

Beyond the built-in A2A messaging (`agent.send()`), the A2A extension
adds broadcast, discovery, and request/reply patterns:

```python
from supervisors.ext.a2a import A2AExtension

a2a = A2AExtension()
agent.use(a2a)

a2a.broadcast(agent, "Hello everyone!")
names = a2a.discover_agents(agent)
```

---

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `Message(sender, recipient, content)` | A message exchanged between agents.  Supports metadata via `set_meta`/`get_meta`. |
| `Supervisor()` | Manages agents and routes messages.  Backed by a tokio async runtime. |
| `Agent(name)` | Base class for all agents.  Subclass and override `handle_message`. |
| `LoopAgent(name, max_iterations=10)` | Agent with iterative reasoning loop.  Override `step()`. |
| `PipelineAgent(name, stages=[])` | Agent with sequential processing stages. |
| `SupervisorAgent(name, router=None)` | Hierarchical agent that delegates to sub-agents. |
| `MultiAgent(name, members=[], max_rounds=10)` | Collaborative group of peer agents. |
| `Extension` | Base class for agent extension plugins. |

### Supervisor Methods

| Method | Description |
|--------|-------------|
| `register(name, handler)` | Register an agent with a message handler. |
| `unregister(name) -> bool` | Remove an agent. Returns `True` if it existed. |
| `send(msg)` | Enqueue a message for delivery. |
| `run_once() -> int` | Deliver all pending messages.  Returns count processed. |
| `dispatch_async() -> int` | Concurrent dispatch via tokio. |
| `agent_names() -> list[str]` | Names of all registered agents. |
| `agent_count() -> int` | Number of registered agents. |
| `pending_count(name) -> int or None` | Queued messages for an agent. |

### Agent Methods

| Method | Description |
|--------|-------------|
| `handle_message(msg)` | Override to define agent behaviour. |
| `register(supervisor)` | Register with a supervisor. |
| `unregister() -> bool` | Remove from supervisor. |
| `send(recipient, content)` | Send a message to another agent (A2A). |
| `use(extension) -> Agent` | Load an extension plugin. |
| `remove_extension(name) -> bool` | Remove an extension. |

---

## Architecture

```
+-------------------------------------------+
|              Python Layer                 |
|                                           |
|  LoopAgent   PipelineAgent               |
|  SupervisorAgent   MultiAgent            |
|  Agent + Extensions (RAG, MCP, ...)      |
+-------------------------------------------+
|            Rust Core (PyO3)               |
|                                           |
|  Message    Supervisor    ToolRegistry    |
|  (tokio async runtime for concurrency)   |
+-------------------------------------------+
```

The Rust core provides:
- **Message routing** with fault-tolerant delivery
- **Async dispatch** via tokio for concurrent agent execution
- **Tool registry** for high-performance tool specification storage

The Python layer provides:
- **Four composable agent patterns** for any business scenario
- **Extension plugin system** for modular capabilities
- **Rich lifecycle hooks** for logging, metrics, and customisation

---

## Contributing

Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md)
before submitting a pull request.

```bash
# Development setup
git clone https://github.com/HsiangNianian/supervisors.rs.git
cd supervisors.rs
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --skip-install
pip install -e . --no-build-isolation

# Run tests
python -m pytest tests/ -v
```

---

## License

[MIT](LICENSE)
