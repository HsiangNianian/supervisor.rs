"""Multi-Agent -- collaborative multi-agent orchestration.

A :class:`MultiAgent` manages a group of peer agents that can communicate
with each other.  Unlike the hierarchical :class:`SupervisorAgent`, all
agents in a multi-agent group are equal peers that collaborate to solve
problems.

Example::

    from supervisor import Agent, MultiAgent, Message, Supervisor

    class Worker(Agent):
        def handle_message(self, msg):
            print(f"[{self.name}] processing: {msg.content}")

    sup = Supervisor()
    group = MultiAgent("team", members=[Worker("alice"), Worker("bob")])
    group.register(sup)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from supervisor._core import Message, Supervisor
from supervisor.agent import Agent

if TYPE_CHECKING:
    pass

# Type alias for a strategy function that decides message routing.
StrategyFunc = Callable[[Message, List[str]], List[str]]


def _broadcast_strategy(msg: Message, members: List[str]) -> List[str]:
    """Default strategy: send the message to all members."""
    return list(members)


def _round_robin_strategy(msg: Message, members: List[str]) -> List[str]:
    """Round-robin strategy: send to one member at a time."""
    if not members:
        return []
    return [members[0]]


class MultiAgent(Agent):
    """Collaborative multi-agent group.

    All member agents share a private :class:`Supervisor` and can
    communicate freely with each other via standard A2A messaging.

    Parameters:
        name: Unique name for the multi-agent group.
        members: Optional initial list of member agents.
        strategy: Optional routing strategy function
            ``(msg, member_names) -> target_names``.  Defaults to
            broadcasting to all members.
        max_rounds: Maximum number of communication rounds before
            forcing termination.  Prevents infinite message loops.
    """

    def __init__(
        self,
        name: str,
        *,
        members: Optional[List[Agent]] = None,
        strategy: Optional[StrategyFunc] = None,
        max_rounds: int = 10,
    ) -> None:
        super().__init__(name)
        self._inner_sup = Supervisor()
        self._members: Dict[str, Agent] = {}
        self._strategy = strategy or _broadcast_strategy
        self.max_rounds = max_rounds

        if members:
            for member in members:
                self.add_member(member)

    # -- member management ---------------------------------------------------

    def add_member(self, agent: Agent) -> "MultiAgent":
        """Add a member agent to the group.

        The member is registered with the shared internal supervisor so
        that it can communicate with other members.  Returns *self* for
        chaining.
        """
        agent.register(self._inner_sup)
        self._members[agent.name] = agent
        return self

    def remove_member(self, name: str) -> bool:
        """Remove a member agent by *name*.

        Returns ``True`` if the member existed and was removed.
        """
        agent = self._members.pop(name, None)
        if agent is not None:
            agent.unregister()
            return True
        return False

    def get_member(self, name: str) -> Optional[Agent]:
        """Retrieve a member agent by *name*, or ``None`` if not found."""
        return self._members.get(name)

    @property
    def member_names(self) -> List[str]:
        """Return the names of all member agents."""
        return list(self._members.keys())

    @property
    def member_count(self) -> int:
        """Return the number of member agents."""
        return len(self._members)

    # -- orchestration -------------------------------------------------------

    def run_rounds(self) -> int:
        """Run communication rounds until no more messages or max_rounds.

        Returns the total number of messages processed across all rounds.
        """
        total = 0
        for _ in range(self.max_rounds):
            processed = self._inner_sup.run_once()
            if processed == 0:
                break
            total += processed
        return total

    # -- hooks ---------------------------------------------------------------

    def on_group_start(self, msg: Message) -> None:
        """Hook called when a new group task begins.

        Override for logging or initialisation.
        """

    def on_group_end(self, msg: Message, total_processed: int) -> None:
        """Hook called after group processing completes.

        Parameters:
            msg: The original incoming message.
            total_processed: Total messages processed across all rounds.
        """

    # -- message handling ----------------------------------------------------

    def handle_message(self, msg: Message) -> None:
        """Distribute the incoming message to members and run rounds.

        The routing strategy determines which members receive the initial
        message.  Members may then communicate freely with each other
        during subsequent rounds.
        """
        self.on_group_start(msg)

        targets = self._strategy(msg, list(self._members.keys()))
        for target_name in targets:
            if target_name in self._members:
                self._inner_sup.send(
                    Message(msg.sender, target_name, msg.content)
                )

        total = self.run_rounds()
        self.on_group_end(msg, total)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        members = ", ".join(self._members) or "none"
        return (
            f"MultiAgent(name={self.name!r}, "
            f"members=[{members}], "
            f"extensions=[{ext_names}])"
        )


__all__ = ["MultiAgent"]
