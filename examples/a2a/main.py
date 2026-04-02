"""A2A Multi-Agent Example — ReAct-style reasoning with supervisor.rs.

Three agents collaborate via a shared Supervisor:

* **main_agent** — orchestrator with ``bash`` tool, delegates to sub-agents.
* **weather_agent** — uses ``get_weather`` (wttr.in) to report weather.
* **search_agent** — uses ``web_search`` (DuckDuckGo) for web queries.

Sub-agents can also delegate to each other.

Usage::

    cd examples/a2a
    cp .env.example .env   # fill in OPENAI_API_KEY
    uv run python main.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from supervisor import Agent, Message, Supervisor

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# Load .env from the same directory as this script.
_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    print(
        "❌  OPENAI_API_KEY is not set.\n"
        "    Copy .env.example → .env and fill in your key."
    )
    sys.exit(1)

client = OpenAI(api_key=_api_key)
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def bash_tool(command: str) -> str:
    """Execute a bash command and return stdout + stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out = result.stdout
        if result.returncode != 0:
            out += f"\n[stderr] {result.stderr}"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30 s limit)"
    except Exception as exc:
        return f"Error: {exc}"


def get_weather(city: str) -> str:
    """Fetch a one-line weather summary from wttr.in (no API key needed)."""
    try:
        resp = httpx.get(
            f"https://wttr.in/{city}",
            params={"format": "3"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        return resp.text.strip()
    except Exception as exc:
        return f"Weather lookup failed for '{city}': {exc}"


def web_search(query: str) -> str:
    """Search the web via DuckDuckGo Instant Answer API (no key needed)."""
    try:
        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1"},
            timeout=10,
            follow_redirects=True,
        )
        resp.raise_for_status()
        data = resp.json()
        parts: list[str] = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for item in data.get("RelatedTopics", [])[:5]:
            if isinstance(item, dict) and "Text" in item:
                parts.append(item["Text"])
        return "\n".join(parts) if parts else f"No results found for '{query}'"
    except Exception as exc:
        return f"Search failed for '{query}': {exc}"


# ---------------------------------------------------------------------------
# OpenAI tool schemas
# ---------------------------------------------------------------------------

_BASH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash shell command and return the output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                },
            },
            "required": ["command"],
        },
    },
}

_DELEGATE_WEATHER: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_weather_agent",
        "description": (
            "Delegate a weather-related question to the weather sub-agent. "
            "Use this when the user asks about weather, temperature, or forecasts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The weather question to ask",
                },
            },
            "required": ["query"],
        },
    },
}

_DELEGATE_SEARCH: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_search_agent",
        "description": (
            "Delegate a web-search question to the search sub-agent. "
            "Use this when the user wants to look up information online."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
}

_GET_WEATHER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city via wttr.in.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g. 'London', 'Tokyo')",
                },
            },
            "required": ["city"],
        },
    },
}

_WEB_SEARCH_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via DuckDuckGo Instant Answer API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
            },
            "required": ["query"],
        },
    },
}

# Tool sets per agent
MAIN_TOOLS = [_BASH_SCHEMA, _DELEGATE_WEATHER, _DELEGATE_SEARCH]
WEATHER_TOOLS = [_GET_WEATHER_SCHEMA, _DELEGATE_SEARCH]
SEARCH_TOOLS = [_WEB_SEARCH_SCHEMA, _DELEGATE_WEATHER]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------


class ReActAgent(Agent):
    """Agent that uses ReAct (Reason + Act) via OpenAI function-calling.

    Each agent has its own system prompt, tool schemas, and tool_map
    (mapping tool name → callable).  Sub-agent delegation is implemented
    as an ordinary tool that calls another agent's ``react`` method.
    """

    def __init__(
        self,
        name: str,
        *,
        system_prompt: str,
        tools: list[dict[str, Any]],
        tool_map: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name)
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_map: dict[str, Any] = tool_map or {}
        self.last_response: str | None = None

    # -- ReAct loop ----------------------------------------------------------

    def react(self, user_message: str, *, max_steps: int = 10) -> str:
        """Run a ReAct reasoning loop and return the final text answer."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for _step in range(max_steps):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=self.tools or None,
                tool_choice="auto" if self.tools else None,
            )

            choice = response.choices[0]
            msg = choice.message
            # Build a serialisable dict for the assistant turn.
            assistant_turn: dict[str, Any] = {"role": "assistant"}
            if msg.content:
                assistant_turn["content"] = msg.content
            if msg.tool_calls:
                assistant_turn["tool_calls"] = [
                    tc.model_dump() for tc in msg.tool_calls
                ]
            messages.append(assistant_turn)

            if not msg.tool_calls:
                # No tools invoked → reasoning is done.
                self.last_response = msg.content or ""
                return self.last_response

            # Execute each tool call and append observations.
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args: dict[str, Any] = json.loads(tc.function.arguments)

                print(f"  [{self.name}] 🔧 {fn_name}({fn_args})")

                if fn_name in self.tool_map:
                    result = str(self.tool_map[fn_name](**fn_args))
                else:
                    result = f"Unknown tool: {fn_name}"

                # Truncate very long results for readability.
                preview = result[:300] + ("…" if len(result) > 300 else "")
                print(f"  [{self.name}] 📋 {preview}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return "(max reasoning steps reached)"

    # -- Supervisor message handler ------------------------------------------

    def handle_message(self, msg: Message) -> None:
        """Handle an incoming supervisor message via a ReAct loop."""
        response = self.react(msg.content)
        self.last_response = response
        # Send the result back to the caller if possible.
        if msg.sender and self.supervisor:
            try:
                self.send(msg.sender, response)
            except (KeyError, RuntimeError):
                pass


# ---------------------------------------------------------------------------
# Delegation helpers
# ---------------------------------------------------------------------------


def _make_delegator(sub_agent: ReActAgent):
    """Return a tool callable that delegates to *sub_agent* via its ReAct loop."""

    def _delegate(query: str) -> str:
        return sub_agent.react(query)

    return _delegate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("🤖  A2A Multi-Agent System (supervisor.rs)")
    print("=" * 55)

    # -- Create Supervisor (backed by tokio) ---------------------------------
    sup = Supervisor()

    # -- Create agents -------------------------------------------------------
    weather_agent = ReActAgent(
        "weather_agent",
        system_prompt=(
            "You are a weather assistant.  Use the get_weather tool to look up "
            "current weather for cities.  Be concise and helpful.  If you need "
            "extra information, you can ask the search agent."
        ),
        tools=WEATHER_TOOLS,
    )

    search_agent = ReActAgent(
        "search_agent",
        system_prompt=(
            "You are a web-search assistant.  Use the web_search tool to find "
            "information online.  Summarise results concisely.  If the user "
            "asks about weather, delegate to the weather agent instead."
        ),
        tools=SEARCH_TOOLS,
    )

    main_agent = ReActAgent(
        "main_agent",
        system_prompt=(
            "You are a helpful assistant with access to three tools:\n"
            "• bash — execute shell commands\n"
            "• ask_weather_agent — delegate weather questions to a specialist\n"
            "• ask_search_agent — delegate web-search questions to a specialist\n\n"
            "Use ReAct reasoning: think step-by-step, use tools when needed, "
            "then give a final answer."
        ),
        tools=MAIN_TOOLS,
    )

    # -- Register with supervisor --------------------------------------------
    weather_agent.register(sup)
    search_agent.register(sup)
    main_agent.register(sup)

    # -- Wire up tool maps (including cross-delegation) ----------------------
    weather_agent.tool_map = {
        "get_weather": get_weather,
        "ask_search_agent": _make_delegator(search_agent),
    }
    search_agent.tool_map = {
        "web_search": web_search,
        "ask_weather_agent": _make_delegator(weather_agent),
    }
    main_agent.tool_map = {
        "bash": bash_tool,
        "ask_weather_agent": _make_delegator(weather_agent),
        "ask_search_agent": _make_delegator(search_agent),
    }

    # -- Register tool specs in Rust ToolRegistry (demonstration) ---------------
    from supervisor._core import ToolRegistry, ToolSpec

    tool_registry = ToolRegistry()
    for name, desc in [
        ("bash", "Execute a bash command"),
        ("get_weather", "Get weather for a city"),
        ("web_search", "Search the web"),
    ]:
        spec = ToolSpec(name, desc)
        handler = main_agent.tool_map.get(name, lambda: None)
        tool_registry.register(spec, handler)

    print(
        f"✅  {sup.agent_count()} agents registered: "
        f"{', '.join(sup.agent_names())}"
    )
    print(f"\nType a message to chat with the main agent.")
    print(f"Type 'quit' or Ctrl-C to exit.\n{'─' * 55}")

    # -- Interactive loop ----------------------------------------------------
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break

            print()
            response = main_agent.react(user_input)
            print(f"\n🤖 Agent: {response}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"❌  Error: {exc}")

    print("\n👋  Goodbye!")


if __name__ == "__main__":
    main()
