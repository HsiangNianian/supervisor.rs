"""Typed message layer on top of the Rust ``_core.Message``.

Provides Pydantic-validated message types, a registry for automatic
deserialization, and a simple type-based message router.

Example::

    from supervisor.typed_message import ChatMessage, TypedRouter

    router = TypedRouter()

    @router.on("chat")
    def handle_chat(msg: ChatMessage) -> None:
        print(f"[{msg.role}] {msg.content}")

    msg = ChatMessage(sender="alice", recipient="bob",
                      content="Hello!", role="user")
    router.dispatch(msg)
"""

from __future__ import annotations

import json
from typing import Any, Callable, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from supervisor._core import Message


# ---------------------------------------------------------------------------
# Base typed message
# ---------------------------------------------------------------------------


class TypedMessage(BaseModel):
    """Base class for all typed messages.

    Wraps the Rust ``Message`` with additional metadata, Pydantic
    validation, and (de)serialization helpers.

    Attributes:
        msg_type: Discriminator string, e.g. ``"chat"``, ``"tool_call"``.
        sender: Name of the sending agent.
        recipient: Name of the target agent.
        content: Free-form text payload.
        metadata: Optional bag of extra key/value pairs.
    """

    msg_type: str = "generic"
    sender: str = ""
    recipient: str = ""
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- serialization -------------------------------------------------------

    def to_json(self) -> str:
        """Serialize the message to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> TypedMessage:
        """Deserialize a JSON string into a ``TypedMessage`` (or subclass).

        Args:
            data: A JSON-encoded string.
        """
        return cls.model_validate_json(data)

    # -- core bridge ---------------------------------------------------------

    def to_core_message(self) -> Message:
        """Convert to the Rust ``_core.Message``.

        The full typed payload is JSON-encoded into the ``content`` field
        of the core message so that no information is lost.
        """
        return Message(self.sender, self.recipient, self.to_json())

    @classmethod
    def from_core_message(
        cls,
        msg: Message,
        msg_type: str = "generic",
    ) -> TypedMessage:
        """Create a ``TypedMessage`` from a Rust ``_core.Message``.

        Args:
            msg: The core message to convert.
            msg_type: The ``msg_type`` tag to assign.
        """
        return cls(
            msg_type=msg_type,
            sender=msg.sender,
            recipient=msg.recipient,
            content=msg.content,
        )

    # -- optional msgpack support --------------------------------------------

    def to_msgpack(self) -> bytes:
        """Serialize the message to MessagePack bytes.

        Raises:
            ImportError: If the ``msgpack`` package is not installed.
        """
        try:
            import msgpack
        except ImportError as exc:
            raise ImportError(
                "msgpack is required for to_msgpack() – "
                "install it with: pip install msgpack"
            ) from exc
        return msgpack.packb(self.model_dump(), use_bin_type=True)

    @classmethod
    def from_msgpack(cls, data: bytes) -> TypedMessage:
        """Deserialize MessagePack bytes into a ``TypedMessage``.

        Args:
            data: MessagePack-encoded bytes.

        Raises:
            ImportError: If the ``msgpack`` package is not installed.
        """
        try:
            import msgpack
        except ImportError as exc:
            raise ImportError(
                "msgpack is required for from_msgpack() – "
                "install it with: pip install msgpack"
            ) from exc
        payload = msgpack.unpackb(data, raw=False)
        return cls.model_validate(payload)


# ---------------------------------------------------------------------------
# Pre-defined message types
# ---------------------------------------------------------------------------


class ChatMessage(TypedMessage):
    """A conversational chat message.

    Attributes:
        role: Speaker role, e.g. ``"user"``, ``"assistant"``, ``"system"``.
    """

    msg_type: str = "chat"
    role: str = "user"


class ToolCallMessage(TypedMessage):
    """A request to invoke a tool.

    Attributes:
        tool_name: Name of the tool to call.
        args: Keyword arguments for the tool.
    """

    msg_type: str = "tool_call"
    tool_name: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResultMessage(TypedMessage):
    """The result returned by a tool invocation.

    Attributes:
        tool_name: Name of the tool that produced this result.
        result: Arbitrary result value.
    """

    msg_type: str = "tool_result"
    tool_name: str = ""
    result: Any = None


class SystemMessage(TypedMessage):
    """A system-level control message."""

    msg_type: str = "system"


class ErrorMessage(TypedMessage):
    """An error notification.

    Attributes:
        error_code: Machine-readable error identifier.
        traceback: Optional formatted traceback string.
    """

    msg_type: str = "error"
    error_code: str = ""
    traceback: Optional[str] = None


# ---------------------------------------------------------------------------
# Message registry (singleton)
# ---------------------------------------------------------------------------


class MessageRegistry:
    """Singleton registry that maps ``msg_type`` strings to classes.

    Built-in message types are registered automatically on first access.
    Third-party code can register additional types via :meth:`register`.

    Example::

        registry = MessageRegistry()
        registry.register("my_type", MyMessage)
        msg = registry.deserialize(json_str)
    """

    _instance: ClassVar[Optional[MessageRegistry]] = None

    def __new__(cls) -> MessageRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._types: dict[str, Type[TypedMessage]] = {}
            cls._instance._register_builtins()
        return cls._instance

    # -- public API ----------------------------------------------------------

    def register(self, msg_type: str, cls: Type[TypedMessage]) -> None:
        """Register *cls* for *msg_type*.

        Args:
            msg_type: The discriminator string.
            cls: The ``TypedMessage`` subclass to associate.
        """
        self._types[msg_type] = cls

    def resolve(self, msg_type: str) -> Type[TypedMessage]:
        """Return the class registered for *msg_type*.

        Args:
            msg_type: The discriminator string.

        Raises:
            KeyError: If *msg_type* has not been registered.
        """
        try:
            return self._types[msg_type]
        except KeyError:
            raise KeyError(
                f"Unknown message type {msg_type!r}. "
                f"Registered types: {list(self._types)}"
            ) from None

    def deserialize(self, json_str: str) -> TypedMessage:
        """Deserialize *json_str*, auto-resolving the concrete type.

        The JSON object must contain a ``"msg_type"`` key.  The
        corresponding class is looked up in the registry and used for
        validation / construction.

        Args:
            json_str: A JSON-encoded typed message.

        Raises:
            KeyError: If the ``msg_type`` value is not registered.
            ValueError: If ``msg_type`` is missing from the JSON.
        """
        raw = json.loads(json_str)
        msg_type = raw.get("msg_type")
        if msg_type is None:
            raise ValueError(
                "JSON payload is missing the required 'msg_type' field"
            )
        cls = self.resolve(msg_type)
        return cls.model_validate(raw)

    # -- internals -----------------------------------------------------------

    def _register_builtins(self) -> None:
        """Seed the registry with the built-in message types."""
        for cls in (
            TypedMessage,
            ChatMessage,
            ToolCallMessage,
            ToolResultMessage,
            SystemMessage,
            ErrorMessage,
        ):
            field = cls.model_fields.get("msg_type")
            key = getattr(field, "default", None) if field else None
            if key is not None:
                self._types[key] = cls

    def _reset(self) -> None:
        """Reset the registry to its initial state (testing helper)."""
        self._types.clear()
        self._register_builtins()


# ---------------------------------------------------------------------------
# Typed router
# ---------------------------------------------------------------------------


class TypedRouter:
    """Routes :class:`TypedMessage` instances by ``msg_type``.

    Register handlers with :meth:`on`, then call :meth:`dispatch` to
    invoke the matching handler.

    Example::

        router = TypedRouter()

        @router.on("chat")
        def on_chat(msg):
            print(msg.content)

        router.dispatch(chat_msg)
    """

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[..., Any]] = {}

    def on(
        self,
        msg_type: str,
        handler: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """Register a handler for *msg_type*.

        Can be used as a plain method call or as a decorator::

            # decorator style
            @router.on("chat")
            def handle(msg): ...

            # direct style
            router.on("chat", handle)

        Args:
            msg_type: The ``msg_type`` discriminator to match.
            handler: The callable to invoke.  When omitted, ``on``
                returns a decorator.

        Returns:
            The handler (unchanged) — so the decorator is transparent.
        """
        if handler is not None:
            self._handlers[msg_type] = handler
            return handler

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._handlers[msg_type] = fn
            return fn

        return _decorator

    def dispatch(self, msg: TypedMessage) -> Any:
        """Dispatch *msg* to its registered handler.

        Args:
            msg: The typed message to route.

        Raises:
            KeyError: If no handler is registered for ``msg.msg_type``.

        Returns:
            Whatever the handler returns.
        """
        try:
            handler = self._handlers[msg.msg_type]
        except KeyError:
            raise KeyError(
                f"No handler registered for msg_type={msg.msg_type!r}"
            ) from None
        return handler(msg)


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    "ChatMessage",
    "ErrorMessage",
    "MessageRegistry",
    "SystemMessage",
    "ToolCallMessage",
    "ToolResultMessage",
    "TypedMessage",
    "TypedRouter",
]
