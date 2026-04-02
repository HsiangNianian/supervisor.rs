"""Type stubs for the Rust ``_core`` extension module.

These stubs provide IDE auto-completion and type checking for the types
exposed by the compiled Rust library.
"""

from typing import Callable, Optional


class Message:
    """A message exchanged between agents."""

    sender: str
    recipient: str
    content: str
    msg_type: str

    def __init__(
        self,
        sender: str,
        recipient: str,
        content: str,
        msg_type: Optional[str] = None,
    ) -> None: ...
    def set_meta(self, key: str, value: str) -> None:
        """Set a metadata key-value pair on the message."""
        ...

    def get_meta(self, key: str) -> Optional[str]:
        """Get a metadata value by key, or None if not present."""
        ...

    def get_all_meta(self) -> dict[str, str]:
        """Return all metadata as a dict."""
        ...

    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class ToolSpec:
    """Specification for a registered tool, stored in Rust."""

    name: str
    description: str
    parameters_json: str

    def __init__(
        self,
        name: str,
        description: str = "",
        parameters_json: str = "{}",
    ) -> None: ...
    def to_dict(self) -> dict[str, object]:
        """Return a dict representation of the tool spec."""
        ...

    def __repr__(self) -> str: ...


class ToolRegistry:
    """Rust-backed registry for tool specifications and Python handlers."""

    def __init__(self) -> None: ...
    def register(self, spec: ToolSpec, handler: Callable[..., object]) -> None:
        """Register a tool with its spec and handler callable."""
        ...

    def unregister(self, name: str) -> bool:
        """Remove a registered tool. Returns True if it existed."""
        ...

    def get_handler(self, name: str) -> Callable[..., object]:
        """Get the handler callable for a tool. Raises KeyError if unknown."""
        ...

    def list_tools(self) -> list[ToolSpec]:
        """Return all registered tool specs."""
        ...

    def get_spec(self, name: str) -> Optional[ToolSpec]:
        """Get the spec for a specific tool, or None."""
        ...

    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        ...

    def tool_count(self) -> int:
        """Return the number of registered tools."""
        ...

    def has_tool(self, name: str) -> bool:
        """Check whether a tool is registered."""
        ...


class Supervisor:
    """Manages agents and routes messages, powered by tokio async runtime."""

    def __init__(self) -> None: ...
    def register(self, name: str, handler: Callable[[Message], None]) -> None:
        """Register an agent with the given name and message-handler callable."""
        ...

    def unregister(self, name: str) -> bool:
        """Remove an agent. Returns True if the agent existed."""
        ...

    def send(self, msg: Message) -> None:
        """Enqueue a message for delivery. Raises KeyError for unknown recipients."""
        ...

    def run_once(self) -> int:
        """Deliver all pending messages synchronously. Returns number processed."""
        ...

    def dispatch_async(self) -> int:
        """Deliver all pending messages using tokio. Returns number processed."""
        ...

    def agent_names(self) -> list[str]:
        """Return the names of all registered agents."""
        ...

    def pending_count(self, name: str) -> Optional[int]:
        """Return the number of queued messages for an agent, or None if unknown."""
        ...

    def agent_count(self) -> int:
        """Return the total number of registered agents."""
        ...

