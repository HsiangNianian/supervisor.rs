"""A2A (Agent-to-Agent) extension.

The :class:`A2AExtension` provides enhanced agent-to-agent communication
features beyond the basic :meth:`Agent.send` built into the agent base class.

While every :class:`~supervisor.agent.Agent` automatically supports basic
A2A via :meth:`~supervisor.agent.Agent.send`, loading this extension adds
broadcast, request/reply patterns, and agent discovery.

Example::

    from supervisor.ext.a2a import A2AExtension

    agent_a.use(A2AExtension())
    agent_b.use(A2AExtension())

    # Broadcast to all agents
    agent_a.extensions["a2a"].broadcast("Hello everyone!")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisor.ext import Extension

if TYPE_CHECKING:
    from supervisor._core import Message
    from supervisor.agent import Agent

# Type alias for reply handler callbacks.
ReplyHandler = Callable[..., Any]


class A2AExtension(Extension):
    """Enhanced Agent-to-Agent communication extension.

    Features beyond the base-class :meth:`Agent.send`:

    * :meth:`broadcast` — send a message to every other agent on the
      supervisor.
    * :meth:`discover_agents` — list all agents on the supervisor.
    * :meth:`request` — send a message and register a reply callback.

    Note:
        Basic A2A (point-to-point messaging) is **automatically** provided
        by the :class:`~supervisor.agent.Agent` base class.  This extension
        is optional and adds higher-level patterns.
    """

    name: str = "a2a"

    def __init__(self) -> None:
        self._reply_callbacks: Dict[str, ReplyHandler] = {}

    # -- broadcasting --------------------------------------------------------

    def broadcast(self, agent: "Agent", content: str) -> int:
        """Send *content* to **every other** registered agent.

        Returns the number of messages enqueued.

        Raises:
            RuntimeError: If the agent is not registered with a supervisor.
        """
        if agent.supervisor is None:
            raise RuntimeError(
                f"Agent '{agent.name}' is not registered with a supervisor."
            )
        count = 0
        for name in agent.supervisor.agent_names():
            if name != agent.name:
                agent.send(name, content)
                count += 1
        return count

    # -- discovery -----------------------------------------------------------

    def discover_agents(self, agent: "Agent") -> List[str]:
        """Return the names of all agents on the same supervisor.

        Raises:
            RuntimeError: If the agent is not registered with a supervisor.
        """
        if agent.supervisor is None:
            raise RuntimeError(
                f"Agent '{agent.name}' is not registered with a supervisor."
            )
        return agent.supervisor.agent_names()

    # -- request / reply -----------------------------------------------------

    def request(
        self,
        agent: "Agent",
        recipient: str,
        content: str,
        reply_handler: ReplyHandler,
    ) -> None:
        """Send a message and register a *reply_handler* for the response.

        The *reply_handler* is stored and can be dispatched when a reply
        from *recipient* arrives (the agent's ``handle_message`` can check
        :meth:`get_reply_handler`).
        """
        self._reply_callbacks[recipient] = reply_handler
        agent.send(recipient, content)

    def get_reply_handler(self, sender: str) -> Optional[ReplyHandler]:
        """Pop and return the reply callback registered for *sender*, if any."""
        return self._reply_callbacks.pop(sender, None)


__all__ = ["A2AExtension"]
