from typing import Callable, Optional


class Message:
    """A message exchanged between agents."""

    sender: str
    recipient: str
    content: str

    def __init__(self, sender: str, recipient: str, content: str) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class Supervisor:
    """Manages a collection of named agents and routes messages between them."""

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
        """Deliver all pending messages. Returns the number processed."""
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
