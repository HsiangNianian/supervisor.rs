"""Supervisor Agent -- hierarchical supervisor with sub-agents.

A :class:`SupervisorAgent` manages a set of named sub-agents, routing
incoming tasks to the appropriate sub-agent and collecting their results.
This pattern supports divide-and-conquer workflows where a parent agent
delegates specialised work to child agents.

Example::

    from supervisor import (
        Agent, SupervisorAgent, Message, Supervisor,
    )

    class Translator(Agent):
        def handle_message(self, msg):
            # simulate translation
            self.result = f"[translated] {msg.content}"

    sup = Supervisor()
    translator = Translator("translator")

    manager = SupervisorAgent("manager")
    manager.add_sub_agent(translator)
    manager.register(sup)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisor._core import Message, Supervisor
from supervisor.agent import Agent

if TYPE_CHECKING:
    pass

# Type alias for a routing function.
RouterFunc = Callable[[Message], Optional[str]]


class SupervisorAgent(Agent):
    """Hierarchical agent that manages and delegates to sub-agents.

    The supervisor agent holds a private :class:`Supervisor` instance to
    manage its sub-agents.  When a message arrives, a routing function
    determines which sub-agent should handle it.

    Parameters:
        name: Unique name for the supervisor agent.
        router: Optional routing function ``(msg) -> sub_agent_name``.
            If not provided, messages are broadcast to all sub-agents.
    """

    def __init__(
        self,
        name: str,
        *,
        router: Optional[RouterFunc] = None,
    ) -> None:
        super().__init__(name)
        self._inner_sup = Supervisor()
        self._sub_agents: Dict[str, Agent] = {}
        self._router = router
        self._results: Dict[str, Any] = {}

    # -- sub-agent management ------------------------------------------------

    def add_sub_agent(self, agent: Agent) -> "SupervisorAgent":
        """Add a sub-agent to this supervisor.

        The sub-agent is registered with the internal supervisor instance.
        Returns *self* for chaining.
        """
        agent.register(self._inner_sup)
        self._sub_agents[agent.name] = agent
        return self

    def remove_sub_agent(self, name: str) -> bool:
        """Remove a sub-agent by *name*.

        Returns ``True`` if the sub-agent existed and was removed.
        """
        agent = self._sub_agents.pop(name, None)
        if agent is not None:
            agent.unregister()
            return True
        return False

    def get_sub_agent(self, name: str) -> Optional[Agent]:
        """Retrieve a sub-agent by *name*, or ``None`` if not found."""
        return self._sub_agents.get(name)

    @property
    def sub_agent_names(self) -> List[str]:
        """Return the names of all sub-agents."""
        return list(self._sub_agents.keys())

    @property
    def sub_agent_count(self) -> int:
        """Return the number of sub-agents."""
        return len(self._sub_agents)

    # -- routing -------------------------------------------------------------

    def route(self, msg: Message) -> Optional[str]:
        """Determine which sub-agent should handle *msg*.

        If a custom router was provided, it is used.  Otherwise returns
        ``None`` to indicate broadcast to all sub-agents.

        Returns:
            Name of the target sub-agent, or ``None`` for broadcast.
        """
        if self._router is not None:
            return self._router(msg)
        return None

    # -- delegation ----------------------------------------------------------

    def delegate(self, sub_agent_name: str, content: str) -> None:
        """Send a message to a specific sub-agent.

        Parameters:
            sub_agent_name: Name of the target sub-agent.
            content: Message content to send.

        Raises:
            KeyError: If no sub-agent with *sub_agent_name* exists.
        """
        if sub_agent_name not in self._sub_agents:
            raise KeyError(
                f"No sub-agent registered with name '{sub_agent_name}'"
            )
        self._inner_sup.send(
            Message(self.name, sub_agent_name, content)
        )

    def broadcast_to_subs(self, content: str) -> int:
        """Broadcast a message to all sub-agents.

        Returns the number of messages enqueued.
        """
        count = 0
        for sub_name in self._sub_agents:
            self._inner_sup.send(Message(self.name, sub_name, content))
            count += 1
        return count

    def run_sub_agents(self) -> int:
        """Dispatch all pending messages to sub-agents.

        Returns the number of messages successfully processed.
        """
        return self._inner_sup.run_once()

    # -- hooks ---------------------------------------------------------------

    def on_delegate(self, msg: Message, target: str) -> None:
        """Hook called before delegating a message to a sub-agent.

        Override for logging, transformation, or validation.
        """

    def on_sub_agents_complete(self, processed: int) -> None:
        """Hook called after all sub-agents finish processing.

        Parameters:
            processed: Number of messages that were successfully processed.
        """

    # -- message handling ----------------------------------------------------

    def handle_message(self, msg: Message) -> None:
        """Route the incoming message to sub-agents and process.

        If the router returns a specific sub-agent name, the message is
        sent only to that sub-agent.  Otherwise it is broadcast to all
        sub-agents.  After dispatching, :meth:`run_sub_agents` is called
        to process the queued messages.
        """
        target = self.route(msg)
        if target is not None:
            self.on_delegate(msg, target)
            self.delegate(target, msg.content)
        else:
            for sub_name in self._sub_agents:
                self.on_delegate(msg, sub_name)
            self.broadcast_to_subs(msg.content)

        processed = self.run_sub_agents()
        self.on_sub_agents_complete(processed)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        subs = ", ".join(self._sub_agents) or "none"
        return (
            f"SupervisorAgent(name={self.name!r}, "
            f"sub_agents=[{subs}], "
            f"extensions=[{ext_names}])"
        )


__all__ = ["SupervisorAgent"]
