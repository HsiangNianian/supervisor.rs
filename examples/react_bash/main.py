"""ReAct Loop with Bash Tool — DeepSeek Reasoner.

Demonstrates the ``LoopAgent`` pattern combined with the
``FunctionCallingExtension`` to build a ReAct-style agent that uses the
DeepSeek ``deepseek-reasoner`` model with a ``bash`` tool.

The agent runs an iterative reasoning loop:
think → call tools → observe → repeat until a final answer is produced.

Features:
    * Persistent conversation history across turns
    * Configurable context window size
    * ``FunctionCallingExtension`` for tool registration

Usage::

    cd examples/react_bash
    uv run python main.py

Environment variables:
    OPENAI_API_KEY  — Required. Your DeepSeek API key.
    OPENAI_BASE_URL — Optional. API endpoint (default: ``https://api.deepseek.com``).
    MODEL           — Optional. Model name (default: ``deepseek-reasoner``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from supervisors import LoopAgent, Message, Supervisor
from supervisors.ext.function_calling import FunctionCallingExtension

# Ensure the package is importable when running from the examples directory.
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root / "src"))

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    print(
        "Error: OPENAI_API_KEY is not set.\n"
        "    Set it in your environment or create a .env file in this directory."
    )
    sys.exit(1)

_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
client = OpenAI(api_key=_api_key, base_url=_base_url)
MODEL = os.getenv("MODEL", "deepseek-reasoner")

# ---------------------------------------------------------------------------
# Bash tool
# ---------------------------------------------------------------------------


def bash(command: str) -> str:
    """Execute a bash shell command and return stdout + stderr."""
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


# ---------------------------------------------------------------------------
# OpenAI tool schemas (built from FunctionCallingExtension specs)
# ---------------------------------------------------------------------------


def _build_openai_tools(fc: FunctionCallingExtension) -> List[Dict[str, Any]]:
    """Convert ``FunctionCallingExtension`` specs to OpenAI tool format."""
    tools: List[Dict[str, Any]] = []
    for spec in fc.list_tools():
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
        )
    return tools


# ---------------------------------------------------------------------------
# ReActLoopAgent
# ---------------------------------------------------------------------------


class ReActLoopAgent(LoopAgent):
    """LoopAgent that uses DeepSeek reasoner with function calling.

    Each ``step()`` call sends the conversation to the LLM.  If the model
    invokes tools the results are appended and the loop continues.  When
    the model responds with text only (no tool calls) the loop terminates.

    Conversation history is preserved across turns up to
    ``max_context_messages`` (system prompt is always included).
    """

    def __init__(
        self,
        name: str,
        *,
        system_prompt: str,
        fc_extension: FunctionCallingExtension,
        max_iterations: int = 15,
        max_context_messages: int = 20,
        max_tool_calls_per_turn: int = 20,
    ) -> None:
        super().__init__(name, max_iterations=max_iterations)
        self.system_prompt = system_prompt
        self.fc = fc_extension
        self.max_context_messages = max_context_messages
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._history: deque[Dict[str, Any]] = deque()

        # Build OpenAI-compatible tool schemas from the extension.
        self._openai_tools = _build_openai_tools(fc_extension)

    def _build_messages(self, user_content: str) -> List[Dict[str, Any]]:
        """Build the message list for the LLM, respecting the context window.

        The system prompt is always included.  Then the most recent
        ``max_context_messages`` entries from history are included,
        followed by the current user message.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self._history:
            messages.extend(self._history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _trim_history(self) -> None:
        """Remove oldest entries if history exceeds ``max_context_messages``."""
        while len(self._history) > self.max_context_messages:
            self._history.popleft()

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = state["messages"]
        tool_calls_this_turn = state.get("tool_calls_this_turn", 0)

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=self._openai_tools or None,
            tool_choice="auto" if self._openai_tools else None,
        )

        choice = response.choices[0]
        msg = choice.message

        assistant_turn: Dict[str, Any] = {"role": "assistant"}
        if msg.content:
            assistant_turn["content"] = msg.content
            print(f"  {self.name}: {msg.content}")

        if msg.tool_calls:
            assistant_turn["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_turn)

        if not msg.tool_calls:
            state["done"] = True
            state["final_answer"] = msg.content or ""
            return state

        for tc in msg.tool_calls:
            if tool_calls_this_turn >= self.max_tool_calls_per_turn:
                print(
                    f"  [{self.name}] max tool calls per turn "
                    f"({self.max_tool_calls_per_turn}) reached, stopping."
                )
                state["done"] = True
                state["final_answer"] = (
                    msg.content
                    or f"(stopped after {self.max_tool_calls_per_turn} tool calls)"
                )
                return state

            fn_name = tc.function.name
            fn_args: Dict[str, Any] = json.loads(tc.function.arguments)

            print(f"  [{self.name}] tool: {fn_name}({fn_args})")

            try:
                result = str(self.fc.call_tool(fn_name, **fn_args))
            except KeyError:
                result = f"Unknown tool: {fn_name}"

            preview = result[:500] + ("…" if len(result) > 500 else "")
            print(f"  [{self.name}] result: {preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
            tool_calls_this_turn += 1

        state["tool_calls_this_turn"] = tool_calls_this_turn
        return state

    def on_loop_start(self, msg: Message, state: Dict[str, Any]) -> None:
        print(f"\n--- Processing message from '{msg.sender}' ---")
        state["messages"] = self._build_messages(msg.content)

    def on_loop_end(self, msg: Message, state: Dict[str, Any], iterations: int) -> None:
        answer = state.get("final_answer", "(no answer)")

        # Persist this turn into history.
        self._history.append({"role": "user", "content": msg.content})
        self._history.append({"role": "assistant", "content": answer})
        self._trim_history()

        print(f"--- Completed in {iterations} iterations ---")
        print(f"\nFinal response to '{msg.sender}':\n  {answer}\n")

        if self.supervisor and msg.sender:
            try:
                self.send(msg.sender, answer)
            except (KeyError, RuntimeError):
                pass

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._history.clear()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a bash shell. "
    "Use the bash tool ONLY when necessary to execute commands and gather information. "
    "Do NOT run exploratory commands for simple greetings or casual questions. "
    "If the user's message is a greeting, simple question, or does not require system interaction, "
    "respond directly WITHOUT using any tools. "
    "When you have enough information, respond with a final answer without using any tools."
)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("ReAct Loop with Bash Tool — DeepSeek Reasoner")
    print(f"Model: {MODEL}")
    print(f"API:   {_base_url}")
    print("=" * 55)

    sup = Supervisor()

    # Set up FunctionCallingExtension with the bash tool.
    fc = FunctionCallingExtension()
    fc.register_tool(
        bash,
        description="Execute a bash shell command and return its output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run",
                },
            },
            "required": ["command"],
        },
    )

    agent = ReActLoopAgent(
        "react_agent",
        system_prompt=SYSTEM_PROMPT,
        fc_extension=fc,
        max_iterations=15,
        max_context_messages=20,
        max_tool_calls_per_turn=10,
    )
    agent.use(fc)
    agent.register(sup)

    print(f"[OK] Agent '{agent.name}' registered with supervisor")
    print(f"[OK] Tools: {', '.join(t.name for t in fc.list_tools())}")
    print(f"[OK] Context window: {agent.max_context_messages} messages")
    print(f"[OK] Max tool calls per turn: {agent.max_tool_calls_per_turn}")
    print("\nType a message to ask the agent.")
    print("Commands: 'quit' to exit, 'clear' to reset conversation history.")
    print(f"{'─' * 55}")

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break
            if user_input.lower() == "clear":
                agent.clear_history()
                print("[OK] Conversation history cleared.")
                continue

            print()
            msg = Message("user", "react_agent", user_input)
            agent.handle_message(msg)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"Error: {exc}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
