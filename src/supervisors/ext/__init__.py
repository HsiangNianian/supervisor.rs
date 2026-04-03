"""Extension plugin system for the supervisor agent framework.

Extensions are loaded onto :class:`~supervisors.agent.Agent` subclasses via
:meth:`Agent.use() <supervisors.agent.Agent.use>` and provide additional
capabilities such as RAG, Function Calling, MCP, Skills, and A2A.

Every extension inherits from :class:`Extension` and can hook into the
agent lifecycle through ``on_load``, ``on_unload``, and ``on_message``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from supervisors._core import Message
    from supervisors.agent import Agent


class Extension:
    """Base class for all agent extensions.

    Subclass this to create a new extension plugin.  Override any of the
    lifecycle hooks below to customise behaviour.

    Subclasses that do not set :attr:`name` explicitly will have it
    auto-derived from the class name.

    Attributes:
        name: Human-readable name used as the key when loading the extension
              onto an agent via :meth:`Agent.use`.
    """

    name: str = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not cls.name:
            cls.name = cls.__name__

    # -- lifecycle hooks -----------------------------------------------------

    def on_load(self, agent: "Agent") -> None:
        """Called when the extension is loaded onto an *agent*.

        Override to perform initialisation (e.g. register extra handlers,
        open connections).
        """

    def on_unload(self, agent: "Agent") -> None:
        """Called when the extension is removed from an *agent*.

        Override to perform clean-up.
        """

    def on_message(self, agent: "Agent", msg: "Message") -> Optional["Message"]:
        """Called **before** the agent's own :meth:`handle_message`.

        Return ``None`` to let the message pass through unchanged, or return a
        *modified* :class:`Message` to replace it.  Raise :exc:`StopIteration`
        to swallow the message entirely (the agent will not see it).
        """
        return None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


__all__ = ["Extension"]
