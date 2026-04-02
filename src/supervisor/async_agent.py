"""Async agent and supervisor for the supervisor agent framework.

Provides :class:`AsyncAgent` and :class:`AsyncSupervisor` — async-native
counterparts of :class:`~supervisor.agent.Agent` and
:class:`~supervisor._core.Supervisor`.  Both sync and async extensions are
supported transparently.

Example::

    import asyncio
    from supervisor import AsyncAgent, AsyncSupervisor, Message

    class GreeterAgent(AsyncAgent):
        async def handle_message(self, msg: Message) -> None:
            print(f"Hello from {self.name}! You said: {msg.content}")

    async def main():
        sup = AsyncSupervisor()
        greeter = GreeterAgent("greeter")
        greeter.register(sup)
        await sup.send(Message("main", "greeter", "Hi!"))
        await sup.run_once()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import inspect
from typing import TYPE_CHECKING, Optional

from supervisor._core import Message
from supervisor._core import Supervisor as _CoreSupervisor
from supervisor.ext import Extension

if TYPE_CHECKING:
    pass


class AsyncSupervisor:
    """Async wrapper around the Rust-backed :class:`~supervisor._core.Supervisor`.

    Delegates sync agent registration and message delivery to the underlying
    :class:`_CoreSupervisor`.  Async agent handlers are managed entirely in
    Python (own message queue + ``await``-based dispatch) while the core
    supervisor is used only for name tracking.

    Attributes:
        _inner: The wrapped synchronous Supervisor instance.
    """

    @staticmethod
    def _noop_handler(msg: Message) -> None:
        """No-op handler used as a shim for async agents in the core supervisor."""

    def __init__(self) -> None:
        self._inner: _CoreSupervisor = _CoreSupervisor()
        self._async_handlers: dict[str, object] = {}
        self._async_queues: dict[str, list[Message]] = {}

    # -- forwarded sync helpers -----------------------------------------------

    def register(self, name: str, handler: object) -> None:
        """Register an agent handler.

        If *handler* is a coroutine function it is stored in a Python-side
        registry (with its own message queue) and a no-op shim is placed in
        the core supervisor for name-tracking.  Sync handlers are registered
        directly with the core supervisor.
        """
        if asyncio.iscoroutinefunction(handler):
            self._async_handlers[name] = handler
            self._async_queues[name] = []
            # Register a no-op shim so agent_names() / agent_count() work.
            self._inner.register(name, self._noop_handler)
        else:
            self._async_handlers.pop(name, None)
            self._async_queues.pop(name, None)
            self._inner.register(name, handler)

    def unregister(self, name: str) -> bool:
        """Remove a registered agent."""
        self._async_handlers.pop(name, None)
        self._async_queues.pop(name, None)
        return self._inner.unregister(name)

    def agent_names(self) -> list[str]:
        """Return names of all registered agents."""
        return self._inner.agent_names()

    def agent_count(self) -> int:
        """Return the number of registered agents."""
        return self._inner.agent_count()

    def pending_count(self, name: str) -> Optional[int]:
        """Return number of queued messages for *name*.

        For async agents the Python-side queue is consulted; for sync agents
        the core supervisor is queried.
        """
        if name in self._async_queues:
            return len(self._async_queues[name])
        return self._inner.pending_count(name)

    # -- async API ------------------------------------------------------------

    async def send(self, msg: Message) -> None:
        """Enqueue a message for delivery (async).

        Raises:
            KeyError: If no agent with ``msg.recipient`` is registered.
        """
        recipient = msg.recipient
        if recipient in self._async_queues:
            self._async_queues[recipient].append(msg)
        else:
            # Delegates to core; raises KeyError for unknown recipients.
            self._inner.send(msg)

    async def run_once(self) -> int:
        """Deliver all pending messages, awaiting async handlers.

        For agents whose handler is a coroutine function the messages are
        drained from the Python-side queue and awaited concurrently.  Sync
        handlers are dispatched via the core supervisor's ``run_once``.

        Returns:
            The total number of messages successfully processed.
        """
        # --- Phase 1: drain and dispatch async agents ------------------------
        async_tasks: list[asyncio.Task[None]] = []
        async_processed = 0
        for name, handler in self._async_handlers.items():
            queue = self._async_queues[name]
            messages = list(queue)
            queue.clear()
            for msg in messages:
                async_tasks.append(
                    asyncio.create_task(
                        self._run_async_handler(name, handler, msg)
                    )
                )
                async_processed += 1

        if async_tasks:
            await asyncio.gather(*async_tasks)

        # --- Phase 2: dispatch sync agents via core --------------------------
        sync_processed = self._inner.run_once()

        return sync_processed + async_processed

    @staticmethod
    async def _run_async_handler(
        name: str, handler: object, msg: Message
    ) -> None:
        """Invoke an async handler with error logging."""
        try:
            await handler(msg)  # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            import sys

            print(
                f"supervisor: agent '{name}' raised an error: {exc}",
                file=sys.stderr,
            )


class AsyncAgent:
    """Async base class for agents.

    Subclass and override :meth:`handle_message` (an async method) to define
    custom behaviour.  Load extensions with :meth:`use` and register with an
    :class:`AsyncSupervisor` via :meth:`register`.

    Both sync :class:`~supervisor.ext.Extension` instances and async
    extensions (whose ``on_message`` is a coroutine) are supported.

    Parameters:
        name: Unique name for the agent within its supervisor.

    Attributes:
        name: The agent's name.
        extensions: Mapping of loaded extension names to instances.
        supervisor: The :class:`AsyncSupervisor` this agent is registered with
                    (or ``None``).
    """

    def __init__(self, name: str) -> None:
        self.name: str = name
        self.extensions: dict[str, Extension] = {}
        self.supervisor: Optional[AsyncSupervisor] = None

    # -- extension management ------------------------------------------------

    def use(self, extension: Extension) -> "AsyncAgent":
        """Load an *extension* plugin onto this agent.

        The extension's :meth:`~Extension.on_load` hook is called immediately.
        If an extension with the same :attr:`~Extension.name` is already
        loaded, it is replaced (after calling :meth:`~Extension.on_unload` on
        the old one).

        Returns *self* so calls can be chained::

            agent.use(RAGExtension()).use(FunctionCallingExtension())
        """
        key = extension.name
        if key in self.extensions:
            self.extensions[key].on_unload(self)
        self.extensions[key] = extension
        extension.on_load(self)
        return self

    def remove_extension(self, name: str) -> bool:
        """Remove a loaded extension by *name*.

        Returns ``True`` if the extension existed and was removed.
        """
        ext = self.extensions.pop(name, None)
        if ext is not None:
            ext.on_unload(self)
            return True
        return False

    # -- message handling ----------------------------------------------------

    async def handle_message(self, msg: Message) -> None:
        """Process an incoming *msg* asynchronously.

        Override in subclasses to implement custom logic.  The default
        implementation does nothing.
        """

    async def _dispatch(self, msg: Message) -> None:
        """Internal async handler registered with the :class:`AsyncSupervisor`.

        Runs each loaded extension's :meth:`~Extension.on_message` hook in
        load order, awaiting async hooks as needed.  If any hook raises
        :exc:`StopIteration` the message is swallowed.  Otherwise the
        (possibly modified) message is forwarded to :meth:`handle_message`.
        """
        current: Message = msg
        for ext in self.extensions.values():
            try:
                if asyncio.iscoroutinefunction(ext.on_message):
                    result = await ext.on_message(self, current)
                else:
                    result = ext.on_message(self, current)
            except StopIteration:
                return  # message swallowed
            if result is not None:
                current = result
        await self.handle_message(current)

    # -- supervisor integration ----------------------------------------------

    def register(self, supervisor: AsyncSupervisor) -> None:
        """Register this agent with an *async supervisor*.

        The agent's internal async dispatcher (which chains extension hooks
        before :meth:`handle_message`) is used as the message handler.
        """
        self.supervisor = supervisor
        supervisor.register(self.name, self._dispatch)

    def unregister(self) -> bool:
        """Remove this agent from its supervisor.

        Returns ``True`` if the agent was registered and successfully removed.
        """
        if self.supervisor is None:
            return False
        result = self.supervisor.unregister(self.name)
        if result:
            self.supervisor = None
        return result

    # -- A2A (agent-to-agent) convenience ------------------------------------

    async def send(self, recipient: str, content: str) -> None:
        """Send a message to another agent on the same supervisor.

        Raises:
            RuntimeError: If the agent is not registered with a supervisor.
        """
        if self.supervisor is None:
            raise RuntimeError(
                f"Agent '{self.name}' is not registered with a supervisor."
            )
        await self.supervisor.send(Message(self.name, recipient, content))

    # -- dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        ext_names = ", ".join(self.extensions) or "none"
        return f"AsyncAgent(name={self.name!r}, extensions=[{ext_names}])"


__all__ = ["AsyncAgent", "AsyncSupervisor"]
