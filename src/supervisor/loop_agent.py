"""Loop Agent -- LLM + Loop pattern.

A :class:`LoopAgent` repeatedly processes messages in a reasoning loop,
calling a user-defined ``step`` function on each iteration until either
a terminal condition is met or a maximum number of iterations is reached.

This pattern is useful for ReAct-style agents, chain-of-thought reasoning,
or any workflow that requires iterative refinement.

Example::

    from supervisor import LoopAgent, Message, Supervisor

    class ReasoningAgent(LoopAgent):
        def step(self, state):
            # Each step refines the state
            state["count"] = state.get("count", 0) + 1
            if state["count"] >= 3:
                state["done"] = True
            return state

    sup = Supervisor()
    agent = ReasoningAgent("reasoner", max_iterations=10)
    agent.register(sup)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from supervisor.agent import Agent

if TYPE_CHECKING:
    from supervisor._core import Message


class LoopAgent(Agent):
    """Agent that executes a reasoning loop on each incoming message.

    Subclass and override :meth:`step` to define the logic for each
    iteration of the loop.  The loop runs until :meth:`should_stop`
    returns ``True`` or ``max_iterations`` is reached.

    Parameters:
        name: Unique name for the agent.
        max_iterations: Maximum number of loop iterations per message.
        state_factory: Optional callable that produces the initial state
            dict for each new message.  Defaults to an empty dict.
    """

    def __init__(
        self,
        name: str,
        *,
        max_iterations: int = 10,
        state_factory: Optional[Callable[["Message"], Dict[str, Any]]] = None,
    ) -> None:
        super().__init__(name)
        self.max_iterations = max_iterations
        self._state_factory = state_factory

    # -- loop interface (override these) -------------------------------------

    def step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single iteration of the loop.

        Override in subclasses to implement custom reasoning logic.
        Return the updated *state* dict.  Set ``state["done"] = True``
        to signal that the loop should terminate.

        Parameters:
            state: The current state dict carried across iterations.

        Returns:
            The updated state dict.
        """
        return state

    def should_stop(self, state: Dict[str, Any], iteration: int) -> bool:
        """Determine whether the loop should stop.

        The default implementation checks for ``state["done"]`` being truthy
        or the iteration count reaching ``max_iterations``.  Override for
        custom termination logic.

        Parameters:
            state: Current state dict.
            iteration: Zero-based iteration index.

        Returns:
            ``True`` if the loop should stop.
        """
        if state.get("done"):
            return True
        if iteration >= self.max_iterations:
            return True
        return False

    def on_loop_start(self, msg: "Message", state: Dict[str, Any]) -> None:
        """Hook called before the loop begins.

        Override to perform setup.  The default implementation is a no-op.
        """

    def on_loop_end(
        self, msg: "Message", state: Dict[str, Any], iterations: int
    ) -> None:
        """Hook called after the loop finishes.

        Override to perform cleanup or send results.  The default
        implementation is a no-op.

        Parameters:
            msg: The original incoming message.
            state: The final state dict.
            iterations: Total number of iterations executed.
        """

    # -- core loop execution -------------------------------------------------

    def _init_state(self, msg: "Message") -> Dict[str, Any]:
        """Build the initial state for a new loop run."""
        if self._state_factory is not None:
            return self._state_factory(msg)
        return {"input": msg.content, "sender": msg.sender}

    def run_loop(self, msg: "Message") -> Dict[str, Any]:
        """Execute the full loop for the given *msg* and return final state.

        This method initialises state, calls :meth:`on_loop_start`, then
        repeatedly invokes :meth:`step` until :meth:`should_stop` returns
        ``True``, and finally calls :meth:`on_loop_end`.

        Returns:
            The final state dict.
        """
        state = self._init_state(msg)
        self.on_loop_start(msg, state)

        iteration = 0
        while not self.should_stop(state, iteration):
            state = self.step(state)
            iteration += 1

        self.on_loop_end(msg, state, iteration)
        return state

    def handle_message(self, msg: "Message") -> None:
        """Process an incoming message by running the loop.

        Subclasses may override this to customise post-loop behaviour
        (e.g. sending the result back to the caller).
        """
        self.run_loop(msg)

    # -- repr ----------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        return (
            f"LoopAgent(name={self.name!r}, "
            f"max_iterations={self.max_iterations}, "
            f"extensions=[{ext_names}])"
        )


__all__ = ["LoopAgent"]
