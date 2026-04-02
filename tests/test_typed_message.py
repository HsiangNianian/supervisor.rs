"""Tests for :mod:`supervisor.typed_message`."""

from __future__ import annotations

import json

import pytest

from supervisor._core import Message
from supervisor.typed_message import (
    ChatMessage,
    ErrorMessage,
    MessageRegistry,
    SystemMessage,
    ToolCallMessage,
    ToolResultMessage,
    TypedMessage,
    TypedRouter,
)


# ---------------------------------------------------------------------------
# TypedMessage base
# ---------------------------------------------------------------------------


class TestTypedMessage:
    """Unit tests for the ``TypedMessage`` base class."""

    def test_defaults(self) -> None:
        msg = TypedMessage()
        assert msg.msg_type == "generic"
        assert msg.sender == ""
        assert msg.recipient == ""
        assert msg.content == ""
        assert msg.metadata == {}

    def test_field_assignment(self) -> None:
        msg = TypedMessage(
            msg_type="custom",
            sender="a",
            recipient="b",
            content="hello",
            metadata={"key": "val"},
        )
        assert msg.msg_type == "custom"
        assert msg.sender == "a"
        assert msg.recipient == "b"
        assert msg.content == "hello"
        assert msg.metadata == {"key": "val"}

    def test_to_json_roundtrip(self) -> None:
        msg = TypedMessage(sender="a", recipient="b", content="hi")
        json_str = msg.to_json()
        restored = TypedMessage.from_json(json_str)
        assert restored == msg

    def test_from_json_with_extra_fields_ignored(self) -> None:
        data = json.dumps(
            {
                "msg_type": "generic",
                "sender": "x",
                "recipient": "y",
                "content": "z",
                "metadata": {},
                "unknown_field": 42,
            }
        )
        # Pydantic v2 ignores extra fields by default.
        msg = TypedMessage.from_json(data)
        assert msg.sender == "x"

    def test_to_core_message(self) -> None:
        msg = TypedMessage(sender="a", recipient="b", content="payload")
        core = msg.to_core_message()
        assert isinstance(core, Message)
        assert core.sender == "a"
        assert core.recipient == "b"
        # Core content is the full JSON payload.
        parsed = json.loads(core.content)
        assert parsed["content"] == "payload"

    def test_from_core_message(self) -> None:
        core = Message("alice", "bob", "hello")
        typed = TypedMessage.from_core_message(core, msg_type="test")
        assert typed.sender == "alice"
        assert typed.recipient == "bob"
        assert typed.content == "hello"
        assert typed.msg_type == "test"

    def test_from_core_message_default_type(self) -> None:
        core = Message("a", "b", "c")
        typed = TypedMessage.from_core_message(core)
        assert typed.msg_type == "generic"

    def test_metadata_preserved_in_json(self) -> None:
        msg = TypedMessage(
            sender="a",
            recipient="b",
            content="c",
            metadata={"priority": 1, "tags": ["urgent"]},
        )
        restored = TypedMessage.from_json(msg.to_json())
        assert restored.metadata == {"priority": 1, "tags": ["urgent"]}

    def test_validation_rejects_wrong_types(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            TypedMessage(sender=123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pre-defined message types
# ---------------------------------------------------------------------------


class TestChatMessage:
    """Tests for ``ChatMessage``."""

    def test_defaults(self) -> None:
        msg = ChatMessage(sender="a", recipient="b", content="hi")
        assert msg.msg_type == "chat"
        assert msg.role == "user"

    def test_custom_role(self) -> None:
        msg = ChatMessage(
            sender="a", recipient="b", content="ok", role="assistant"
        )
        assert msg.role == "assistant"

    def test_json_roundtrip(self) -> None:
        msg = ChatMessage(
            sender="a", recipient="b", content="hi", role="system"
        )
        restored = ChatMessage.from_json(msg.to_json())
        assert restored == msg
        assert restored.role == "system"


class TestToolCallMessage:
    """Tests for ``ToolCallMessage``."""

    def test_defaults(self) -> None:
        msg = ToolCallMessage(sender="a", recipient="b", content="")
        assert msg.msg_type == "tool_call"
        assert msg.tool_name == ""
        assert msg.args == {}

    def test_with_args(self) -> None:
        msg = ToolCallMessage(
            sender="a",
            recipient="b",
            content="call",
            tool_name="search",
            args={"query": "foo"},
        )
        assert msg.tool_name == "search"
        assert msg.args == {"query": "foo"}

    def test_json_roundtrip(self) -> None:
        msg = ToolCallMessage(
            sender="a",
            recipient="b",
            content="",
            tool_name="calc",
            args={"x": 1, "y": 2},
        )
        restored = ToolCallMessage.from_json(msg.to_json())
        assert restored == msg


class TestToolResultMessage:
    """Tests for ``ToolResultMessage``."""

    def test_defaults(self) -> None:
        msg = ToolResultMessage(sender="a", recipient="b", content="")
        assert msg.msg_type == "tool_result"
        assert msg.tool_name == ""
        assert msg.result is None

    def test_with_result(self) -> None:
        msg = ToolResultMessage(
            sender="a",
            recipient="b",
            content="",
            tool_name="calc",
            result={"answer": 42},
        )
        assert msg.result == {"answer": 42}

    def test_json_roundtrip(self) -> None:
        msg = ToolResultMessage(
            sender="a",
            recipient="b",
            content="done",
            tool_name="search",
            result=[1, 2, 3],
        )
        restored = ToolResultMessage.from_json(msg.to_json())
        assert restored == msg


class TestSystemMessage:
    """Tests for ``SystemMessage``."""

    def test_type(self) -> None:
        msg = SystemMessage(sender="sys", recipient="all", content="reboot")
        assert msg.msg_type == "system"

    def test_json_roundtrip(self) -> None:
        msg = SystemMessage(
            sender="sys", recipient="agent", content="shutdown"
        )
        restored = SystemMessage.from_json(msg.to_json())
        assert restored == msg


class TestErrorMessage:
    """Tests for ``ErrorMessage``."""

    def test_defaults(self) -> None:
        msg = ErrorMessage(sender="a", recipient="b", content="fail")
        assert msg.msg_type == "error"
        assert msg.error_code == ""
        assert msg.traceback is None

    def test_with_traceback(self) -> None:
        msg = ErrorMessage(
            sender="a",
            recipient="b",
            content="oops",
            error_code="E001",
            traceback="Traceback ...",
        )
        assert msg.error_code == "E001"
        assert msg.traceback == "Traceback ..."

    def test_json_roundtrip(self) -> None:
        msg = ErrorMessage(
            sender="a",
            recipient="b",
            content="err",
            error_code="E404",
            traceback="line 1\nline 2",
        )
        restored = ErrorMessage.from_json(msg.to_json())
        assert restored == msg


# ---------------------------------------------------------------------------
# MessageRegistry
# ---------------------------------------------------------------------------


class TestMessageRegistry:
    """Tests for the ``MessageRegistry`` singleton."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self) -> None:
        """Ensure a clean registry for every test."""
        MessageRegistry()._reset()

    def test_singleton(self) -> None:
        a = MessageRegistry()
        b = MessageRegistry()
        assert a is b

    def test_resolve_builtin(self) -> None:
        reg = MessageRegistry()
        assert reg.resolve("chat") is ChatMessage
        assert reg.resolve("tool_call") is ToolCallMessage
        assert reg.resolve("tool_result") is ToolResultMessage
        assert reg.resolve("system") is SystemMessage
        assert reg.resolve("error") is ErrorMessage
        assert reg.resolve("generic") is TypedMessage

    def test_resolve_unknown_raises(self) -> None:
        reg = MessageRegistry()
        with pytest.raises(KeyError, match="Unknown message type"):
            reg.resolve("nonexistent")

    def test_register_custom_type(self) -> None:
        class PingMessage(TypedMessage):
            msg_type: str = "ping"

        reg = MessageRegistry()
        reg.register("ping", PingMessage)
        assert reg.resolve("ping") is PingMessage

    def test_deserialize_chat(self) -> None:
        msg = ChatMessage(
            sender="a", recipient="b", content="hi", role="user"
        )
        json_str = msg.to_json()
        reg = MessageRegistry()
        restored = reg.deserialize(json_str)
        assert isinstance(restored, ChatMessage)
        assert restored.role == "user"

    def test_deserialize_error(self) -> None:
        msg = ErrorMessage(
            sender="a",
            recipient="b",
            content="oops",
            error_code="E500",
        )
        reg = MessageRegistry()
        restored = reg.deserialize(msg.to_json())
        assert isinstance(restored, ErrorMessage)
        assert restored.error_code == "E500"

    def test_deserialize_missing_msg_type(self) -> None:
        reg = MessageRegistry()
        with pytest.raises(ValueError, match="missing"):
            reg.deserialize('{"sender": "a"}')

    def test_deserialize_unknown_type(self) -> None:
        reg = MessageRegistry()
        with pytest.raises(KeyError, match="Unknown"):
            reg.deserialize('{"msg_type": "unknown"}')


# ---------------------------------------------------------------------------
# TypedRouter
# ---------------------------------------------------------------------------


class TestTypedRouter:
    """Tests for the ``TypedRouter``."""

    def test_direct_registration(self) -> None:
        router = TypedRouter()
        results: list[str] = []
        router.on("chat", lambda msg: results.append(msg.content))
        router.dispatch(
            ChatMessage(sender="a", recipient="b", content="hello")
        )
        assert results == ["hello"]

    def test_decorator_registration(self) -> None:
        router = TypedRouter()
        calls: list[TypedMessage] = []

        @router.on("system")
        def _handler(msg: TypedMessage) -> None:
            calls.append(msg)

        msg = SystemMessage(sender="sys", recipient="all", content="ping")
        router.dispatch(msg)
        assert len(calls) == 1
        assert calls[0] is msg

    def test_handler_return_value(self) -> None:
        router = TypedRouter()
        router.on("chat", lambda msg: msg.content.upper())
        result = router.dispatch(
            ChatMessage(sender="a", recipient="b", content="hi")
        )
        assert result == "HI"

    def test_dispatch_no_handler(self) -> None:
        router = TypedRouter()
        with pytest.raises(KeyError, match="No handler"):
            router.dispatch(
                TypedMessage(msg_type="unregistered", sender="a")
            )

    def test_multiple_types(self) -> None:
        router = TypedRouter()
        log: list[str] = []
        router.on("chat", lambda m: log.append("chat"))
        router.on("error", lambda m: log.append("error"))

        router.dispatch(
            ChatMessage(sender="a", recipient="b", content="")
        )
        router.dispatch(
            ErrorMessage(sender="a", recipient="b", content="e")
        )
        assert log == ["chat", "error"]

    def test_decorator_preserves_function(self) -> None:
        router = TypedRouter()

        @router.on("chat")
        def my_handler(msg: TypedMessage) -> str:
            return "ok"

        # The original function reference should be unchanged.
        assert my_handler(ChatMessage(sender="a", recipient="b", content="")) == "ok"


# ---------------------------------------------------------------------------
# Core bridge round-trip
# ---------------------------------------------------------------------------


class TestCoreBridge:
    """Test TypedMessage ↔ _core.Message conversion fidelity."""

    def test_typed_to_core_and_back(self) -> None:
        original = ChatMessage(
            sender="alice",
            recipient="bob",
            content="hi",
            role="assistant",
            metadata={"seq": 1},
        )
        core = original.to_core_message()

        # Recover via the registry (auto-resolves to ChatMessage).
        reg = MessageRegistry()
        restored = reg.deserialize(core.content)
        assert isinstance(restored, ChatMessage)
        assert restored.sender == "alice"
        assert restored.role == "assistant"
        assert restored.metadata == {"seq": 1}

    def test_plain_core_message_wrapping(self) -> None:
        core = Message("x", "y", "raw text")
        typed = TypedMessage.from_core_message(core, msg_type="chat")
        assert typed.content == "raw text"
        assert typed.msg_type == "chat"


# ---------------------------------------------------------------------------
# msgpack (optional)
# ---------------------------------------------------------------------------


class TestMsgpack:
    """Tests for optional msgpack (de)serialization."""

    def test_msgpack_import_error(self) -> None:
        """When msgpack is not installed, a helpful error is raised."""
        msg = TypedMessage(sender="a", content="b")
        with pytest.raises(ImportError, match="msgpack"):
            msg.to_msgpack()
        with pytest.raises(ImportError, match="msgpack"):
            TypedMessage.from_msgpack(b"")
