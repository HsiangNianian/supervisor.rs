"""A2A Plan/Todo Multi-Agent Example — DeepSeek Reasoner.

A main planning agent with three sub-agents, using the supervisors SDK
properly:

* **planner_agent** — ``LoopAgent`` with ``todo`` + ``send_task`` tools
* **write_file_agent** — ``LoopAgent`` with ``write_file`` tool
* **read_file_agent** — ``LoopAgent`` with ``read_file`` tool
* **shell_agent** — ``LoopAgent`` with ``pwd`` + ``ls`` tools

All agents use the DeepSeek ``deepseek-reasoner`` model via the OpenAI SDK.
Agent-to-agent communication uses ``Agent.send()`` through the shared
``Supervisor``.

Usage::

    cd examples/a2a_plan_todo
    cp .env.example .env   # fill in OPENAI_API_KEY
    uv run python main.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

from supervisors import LoopAgent, Message, Supervisor
from supervisors.ext.function_calling import FunctionCallingExtension

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")

_api_key = os.getenv("OPENAI_API_KEY", "")
if not _api_key:
    print(
        "Error: OPENAI_API_KEY is not set.\n"
        "    Copy .env.example -> .env and fill in your key."
    )
    sys.exit(1)

_base_url = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")
client = OpenAI(api_key=_api_key, base_url=_base_url)
MODEL = os.getenv("MODEL", "deepseek-reasoner")

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_todo_list: list[dict[str, str]] = []

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def todo(action: str, item: str = "", index: int = 0) -> str:
    """Manage a todo list. Actions: add, list, complete, remove."""
    action = action.lower().strip()
    if action == "add":
        if not item:
            return "Error: 'item' is required for 'add' action."
        _todo_list.append({"task": item, "status": "pending"})
        return f"Added todo: {item}"
    elif action == "list":
        if not _todo_list:
            return "Todo list is empty."
        lines = []
        for i, t in enumerate(_todo_list):
            lines.append(f"  [{i}] [{t['status']}] {t['task']}")
        return "Todo list:\n" + "\n".join(lines)
    elif action == "complete":
        if 0 <= index < len(_todo_list):
            _todo_list[index]["status"] = "done"
            return f"Marked todo '{_todo_list[index]['task']}' as done."
        return f"Error: index {index} out of range."
    elif action == "remove":
        if 0 <= index < len(_todo_list):
            removed = _todo_list.pop(index)
            return f"Removed todo: {removed['task']}"
        return f"Error: index {index} out of range."
    else:
        return f"Error: unknown action '{action}'. Use: add, list, complete, remove."


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as exc:
        return f"Error writing file: {exc}"


def read_file(path: str) -> str:
    """Read content from a file."""
    try:
        return Path(path).read_text()
    except Exception as exc:
        return f"Error reading file: {exc}"


def pwd() -> str:
    """Return the current working directory."""
    return str(Path.cwd())


def ls(path: str = ".") -> str:
    """List directory contents."""
    try:
        return "\n".join(sorted(os.listdir(path)))
    except Exception as exc:
        return f"Error listing directory: {exc}"


# ---------------------------------------------------------------------------
# OpenAI tool schemas (for LLM function calling)
# ---------------------------------------------------------------------------

_TODO_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "todo",
        "description": (
            "Manage a todo list. "
            "Actions: 'add' (requires 'item'), 'list', 'complete' (requires 'index'), 'remove' (requires 'index')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: add, list, complete, remove",
                },
                "item": {
                    "type": "string",
                    "description": "Todo item text (required for 'add')",
                },
                "index": {
                    "type": "integer",
                    "description": "Index in the list (required for 'complete' and 'remove')",
                },
            },
            "required": ["action"],
        },
    },
}

_SEND_TASK_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "send_task",
        "description": "Send a task message to a sub-agent via A2A communication.",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {
                    "type": "string",
                    "description": "Target agent name (write_file_agent, read_file_agent, shell_agent)",
                },
                "message": {
                    "type": "string",
                    "description": "The task message to send",
                },
            },
            "required": ["recipient", "message"],
        },
    },
}

_WRITE_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write content to a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to write to"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    },
}

_READ_FILE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read and return the content of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    },
}

_PWD_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "pwd",
        "description": "Return the current working directory.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_LS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ls",
        "description": "List the contents of a directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (default: '.')",
                },
            },
            "required": [],
        },
    },
}

# ---------------------------------------------------------------------------
# LLMReActAgent — LoopAgent + OpenAI function calling
# ---------------------------------------------------------------------------


class LLMReActAgent(LoopAgent):
    """LoopAgent that uses OpenAI function-calling with FunctionCallingExtension.

    Each step() calls the LLM. If tool_calls are returned, they are executed
    via the FunctionCallingExtension. If no tool_calls, the loop terminates.
    """

    def __init__(
        self,
        name: str,
        *,
        system_prompt: str,
        fc_extension: FunctionCallingExtension,
        openai_tools: List[Dict[str, Any]],
        max_iterations: int = 15,
        max_context_messages: int = 20,
        max_tool_calls_per_turn: int = 20,
    ) -> None:
        super().__init__(name, max_iterations=max_iterations)
        self.system_prompt = system_prompt
        self.fc = fc_extension
        self._openai_tools = openai_tools
        self.max_context_messages = max_context_messages
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self._history: deque[Dict[str, Any]] = deque()
        self.last_response: str = ""

    def _build_messages(self, user_content: str) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
        ]
        if self._history:
            messages.extend(list(self._history)[-self.max_context_messages :])
        messages.append({"role": "user", "content": user_content})
        return messages

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        messages: List[Dict[str, Any]] = state["messages"]
        tool_calls_count = state.get("tool_calls_count", 0)

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
            self.last_response = state["final_answer"]
            return state

        for tc in msg.tool_calls:
            if tool_calls_count >= self.max_tool_calls_per_turn:
                print(f"  [{self.name}] max tool calls per turn reached, stopping.")
                state["done"] = True
                state["final_answer"] = (
                    msg.content
                    or f"(stopped after {self.max_tool_calls_per_turn} tool calls)"
                )
                self.last_response = state["final_answer"]
                return state

            fn_name = tc.function.name
            fn_args: Dict[str, Any] = json.loads(tc.function.arguments)
            print(f"  [{self.name}] tool: {fn_name}({fn_args})")

            try:
                result = str(self.fc.call_tool(fn_name, **fn_args))
            except KeyError:
                result = f"Unknown tool: {fn_name}"

            preview = result[:300] + ("..." if len(result) > 300 else "")
            print(f"  [{self.name}] result: {preview}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )
            tool_calls_count += 1

        state["tool_calls_count"] = tool_calls_count
        return state

    def on_loop_start(self, msg: Message, state: Dict[str, Any]) -> None:
        print(f"\n--- [{self.name}] processing message from '{msg.sender}' ---")
        state["messages"] = self._build_messages(msg.content)
        state["tool_calls_count"] = 0

    def on_loop_end(self, msg: Message, state: Dict[str, Any], iterations: int) -> None:
        answer = state.get("final_answer", "(no answer)")
        self.last_response = answer

        self._history.append({"role": "user", "content": msg.content})
        self._history.append({"role": "assistant", "content": answer})
        while len(self._history) > self.max_context_messages * 2:
            self._history.popleft()

        print(f"--- [{self.name}] completed in {iterations} iterations ---")

        if msg.sender and self.supervisor:
            try:
                self.send(msg.sender, answer)
            except (KeyError, RuntimeError):
                pass

    def clear_history(self) -> None:
        self._history.clear()


# ---------------------------------------------------------------------------
# A2A send_task tool
# ---------------------------------------------------------------------------


def make_send_task_tool(sup: Supervisor, planner_agent: LLMReActAgent):
    """Create a send_task tool that uses Agent.send() for A2A communication.

    The tool receives (recipient, message), wraps it in a Message, sends it
    through the supervisor via Agent.send(), then runs the supervisor to
    process the sub-agent and returns its response.
    """

    def send_task(recipient: str, message: str) -> str:
        preview = message[:80] + ("..." if len(message) > 80 else "")
        print(f'    -> send_task: [{recipient}] "{preview}"')

        # Use the SDK's Agent.send() which wraps Message and routes via supervisor
        planner_agent.send(recipient, message)

        # Drive the supervisor to process the sub-agent's response
        sup.run_once()

        return planner_agent.last_response or "(no response)"

    return send_task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("A2A Plan/Todo Multi-Agent System (supervisors.rs)")
    print(f"Model: {MODEL}")
    print(f"API:   {_base_url}")
    print("=" * 55)

    sup = Supervisor()

    # -- FunctionCallingExtension for each agent --

    # Planner: todo + send_task
    planner_fc = FunctionCallingExtension()
    planner_fc.register_tool(
        todo,
        description="Manage a todo list (add, list, complete, remove).",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "One of: add, list, complete, remove",
                },
                "item": {
                    "type": "string",
                    "description": "Todo item text (required for 'add')",
                },
                "index": {
                    "type": "integer",
                    "description": "Index (required for 'complete'/'remove')",
                },
            },
            "required": ["action"],
        },
    )

    # Write file agent
    write_fc = FunctionCallingExtension()
    write_fc.register_tool(
        write_file,
        description="Write content to a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"},
            },
            "required": ["path", "content"],
        },
    )

    # Read file agent
    read_fc = FunctionCallingExtension()
    read_fc.register_tool(
        read_file,
        description="Read content from a file.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
            },
            "required": ["path"],
        },
    )

    # Shell agent
    shell_fc = FunctionCallingExtension()
    shell_fc.register_tool(
        pwd,
        description="Return the current working directory.",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    shell_fc.register_tool(
        ls,
        description="List directory contents.",
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (default: '.')",
                },
            },
            "required": [],
        },
    )

    # -- Create agents --

    planner_agent = LLMReActAgent(
        "planner_agent",
        system_prompt=(
            "You are a planning assistant that manages tasks and delegates to specialists.\n"
            "Tools available:\n"
            "  - todo: manage a todo list (add, list, complete, remove tasks)\n"
            "  - send_task: send a message to a sub-agent via A2A communication.\n"
            "    Available recipients: write_file_agent, read_file_agent, shell_agent\n\n"
            "Use ReAct reasoning: think step-by-step, use the todo tool to plan, "
            "delegate to sub-agents via send_task when needed, then give a final answer."
        ),
        fc_extension=planner_fc,
        openai_tools=[_TODO_SCHEMA, _SEND_TASK_SCHEMA],
        max_iterations=20,
        max_tool_calls_per_turn=15,
    )
    planner_agent.use(planner_fc)

    write_file_agent = LLMReActAgent(
        "write_file_agent",
        system_prompt="You are a file-writing assistant. Use the write_file tool to write content to files. Be concise.",
        fc_extension=write_fc,
        openai_tools=[_WRITE_FILE_SCHEMA],
    )
    write_file_agent.use(write_fc)

    read_file_agent = LLMReActAgent(
        "read_file_agent",
        system_prompt="You are a file-reading assistant. Use the read_file tool to read and display file contents.",
        fc_extension=read_fc,
        openai_tools=[_READ_FILE_SCHEMA],
    )
    read_file_agent.use(read_fc)

    shell_agent = LLMReActAgent(
        "shell_agent",
        system_prompt="You are a shell assistant. Use pwd and ls to show directory information.",
        fc_extension=shell_fc,
        openai_tools=[_PWD_SCHEMA, _LS_SCHEMA],
    )
    shell_agent.use(shell_fc)

    # -- Register all agents with the shared supervisor --
    planner_agent.register(sup)
    write_file_agent.register(sup)
    read_file_agent.register(sup)
    shell_agent.register(sup)

    # -- Wire up send_task tool (needs sup + planner_agent references) --
    send_task_fn = make_send_task_tool(sup, planner_agent)
    planner_fc.register_tool(
        send_task_fn,
        description="Send a task message to a sub-agent via A2A.",
        parameters={
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Target agent name"},
                "message": {
                    "type": "string",
                    "description": "The task message to send",
                },
            },
            "required": ["recipient", "message"],
        },
    )

    print(f"[OK] {sup.agent_count()} agents registered: {', '.join(sup.agent_names())}")
    print("[OK] Planner: todo + send_task (A2A via Agent.send)")
    print("[OK] Sub-agents: write_file_agent, read_file_agent, shell_agent")
    print("\nType a message to the planner agent.")
    print("Commands: 'quit' to exit, 'todos' to show current todo list.")
    print(f"{'-' * 55}")

    # -- Interactive loop --
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                break
            if user_input.lower() == "todos":
                print(todo("list"))
                continue

            print()
            msg = Message("user", "planner_agent", user_input)
            planner_agent.handle_message(msg)
            sup.run_once()
            print(f"\nAgent: {planner_agent.last_response}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            print(f"Error: {exc}")

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
