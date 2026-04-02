"""Tests for the async agent and supervisor module."""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from supervisor._core import Message
from supervisor.async_agent import AsyncAgent, AsyncSupervisor
from supervisor.ext import Extension


# ---------------------------------------------------------------------------
# AsyncSupervisor
# ---------------------------------------------------------------------------


class TestAsyncSupervisor:
    """Tests for the :class:`AsyncSupervisor` wrapper."""

    @pytest.mark.asyncio
    async def test_create(self) -> None:
        sup = AsyncSupervisor()
        assert sup.agent_count() == 0

    @pytest.mark.asyncio
    async def test_register_sync_handler(self) -> None:
        received: list[str] = []
        sup = AsyncSupervisor()
        sup.register("echo", lambda msg: received.append(msg.content))
        assert "echo" in sup.agent_names()

        await sup.send(Message("src", "echo", "hi"))
        await sup.run_once()
        assert received == ["hi"]

    @pytest.mark.asyncio
    async def test_register_async_handler(self) -> None:
        received: list[str] = []

        async def handler(msg: Message) -> None:
            received.append(msg.content)

        sup = AsyncSupervisor()
        sup.register("echo", handler)
        assert "echo" in sup.agent_names()

        await sup.send(Message("src", "echo", "hello"))
        await sup.run_once()
        assert received == ["hello"]

    @pytest.mark.asyncio
    async def test_unregister(self) -> None:
        sup = AsyncSupervisor()
        sup.register("a", lambda msg: None)
        assert sup.unregister("a") is True
        assert "a" not in sup.agent_names()

    @pytest.mark.asyncio
    async def test_unregister_nonexistent(self) -> None:
        sup = AsyncSupervisor()
        assert sup.unregister("nope") is False

    @pytest.mark.asyncio
    async def test_send_unknown_recipient_raises(self) -> None:
        sup = AsyncSupervisor()
        with pytest.raises(KeyError, match="nobody"):
            await sup.send(Message("src", "nobody", "hi"))

    @pytest.mark.asyncio
    async def test_pending_count(self) -> None:
        sup = AsyncSupervisor()
        sup.register("a", lambda msg: None)
        assert sup.pending_count("a") == 0
        await sup.send(Message("src", "a", "hi"))
        assert sup.pending_count("a") == 1
        await sup.run_once()
        assert sup.pending_count("a") == 0

    @pytest.mark.asyncio
    async def test_pending_count_unknown(self) -> None:
        sup = AsyncSupervisor()
        assert sup.pending_count("nope") is None

    @pytest.mark.asyncio
    async def test_run_once_returns_count(self) -> None:
        sup = AsyncSupervisor()
        sup.register("a", lambda msg: None)
        await sup.send(Message("x", "a", "1"))
        await sup.send(Message("x", "a", "2"))
        count = await sup.run_once()
        assert count == 2

    @pytest.mark.asyncio
    async def test_async_handler_error_is_logged(self, capsys: pytest.CaptureFixture[str]) -> None:
        async def bad_handler(msg: Message) -> None:
            raise ValueError("boom")

        sup = AsyncSupervisor()
        sup.register("bad", bad_handler)
        await sup.send(Message("x", "bad", "hi"))
        count = await sup.run_once()
        # The message was consumed (processed) even though handler errored
        assert count == 1
        captured = capsys.readouterr()
        assert "boom" in captured.err

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_handlers(self) -> None:
        sync_received: list[str] = []
        async_received: list[str] = []

        async def async_handler(msg: Message) -> None:
            async_received.append(msg.content)

        sup = AsyncSupervisor()
        sup.register("sync_agent", lambda msg: sync_received.append(msg.content))
        sup.register("async_agent", async_handler)
        await sup.send(Message("x", "sync_agent", "sync_msg"))
        await sup.send(Message("x", "async_agent", "async_msg"))
        await sup.run_once()
        assert sync_received == ["sync_msg"]
        assert async_received == ["async_msg"]

    @pytest.mark.asyncio
    async def test_multiple_async_messages(self) -> None:
        received: list[str] = []

        async def handler(msg: Message) -> None:
            received.append(msg.content)

        sup = AsyncSupervisor()
        sup.register("a", handler)
        await sup.send(Message("x", "a", "m1"))
        await sup.send(Message("x", "a", "m2"))
        await sup.send(Message("x", "a", "m3"))
        await sup.run_once()
        assert sorted(received) == ["m1", "m2", "m3"]


# ---------------------------------------------------------------------------
# AsyncAgent base class
# ---------------------------------------------------------------------------


class TestAsyncAgent:
    """Tests for the :class:`AsyncAgent` base class."""

    @pytest.mark.asyncio
    async def test_create(self) -> None:
        agent = AsyncAgent("test")
        assert agent.name == "test"
        assert agent.extensions == {}
        assert agent.supervisor is None

    @pytest.mark.asyncio
    async def test_register_with_supervisor(self) -> None:
        sup = AsyncSupervisor()
        agent = AsyncAgent("myagent")
        agent.register(sup)
        assert "myagent" in sup.agent_names()
        assert agent.supervisor is sup

    @pytest.mark.asyncio
    async def test_unregister(self) -> None:
        sup = AsyncSupervisor()
        agent = AsyncAgent("myagent")
        agent.register(sup)
        result = agent.unregister()
        assert result is True
        assert "myagent" not in sup.agent_names()
        assert agent.supervisor is None

    @pytest.mark.asyncio
    async def test_unregister_without_supervisor(self) -> None:
        agent = AsyncAgent("myagent")
        assert agent.unregister() is False

    @pytest.mark.asyncio
    async def test_subclass_handle_message(self) -> None:
        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("sub")
        agent.register(sup)
        await sup.send(Message("src", "sub", "hello"))
        await sup.run_once()
        assert received == ["hello"]

    @pytest.mark.asyncio
    async def test_a2a_send(self) -> None:
        received: list[tuple[str, str]] = []

        class Receiver(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append((msg.sender, msg.content))

        sup = AsyncSupervisor()
        sender = AsyncAgent("sender")
        receiver = Receiver("receiver")
        sender.register(sup)
        receiver.register(sup)
        await sender.send("receiver", "hi from sender")
        await sup.run_once()
        assert received == [("sender", "hi from sender")]

    @pytest.mark.asyncio
    async def test_a2a_send_without_supervisor_raises(self) -> None:
        agent = AsyncAgent("lonely")
        with pytest.raises(RuntimeError, match="not registered"):
            await agent.send("nobody", "msg")

    @pytest.mark.asyncio
    async def test_repr(self) -> None:
        agent = AsyncAgent("test")
        r = repr(agent)
        assert "AsyncAgent" in r
        assert "test" in r

    @pytest.mark.asyncio
    async def test_use_returns_self(self) -> None:
        agent = AsyncAgent("a")
        result = agent.use(Extension())
        assert result is agent

    @pytest.mark.asyncio
    async def test_chained_use(self) -> None:
        ext1 = Extension()
        ext1.name = "ext1"
        ext2 = Extension()
        ext2.name = "ext2"
        agent = AsyncAgent("a").use(ext1).use(ext2)
        assert "ext1" in agent.extensions
        assert "ext2" in agent.extensions


# ---------------------------------------------------------------------------
# Sync extension support on AsyncAgent
# ---------------------------------------------------------------------------


class TestAsyncAgentWithSyncExtensions:
    """Sync extensions work correctly on :class:`AsyncAgent`."""

    @pytest.mark.asyncio
    async def test_sync_extension_on_message(self) -> None:
        class UpperExt(Extension):
            name = "upper"

            def on_message(self, agent: object, msg: Message) -> Message:
                return Message(msg.sender, msg.recipient, msg.content.upper())

        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("a")
        agent.use(UpperExt())
        agent.register(sup)
        await sup.send(Message("x", "a", "hello"))
        await sup.run_once()
        assert received == ["HELLO"]

    @pytest.mark.asyncio
    async def test_sync_extension_swallows_message(self) -> None:
        class SwallowExt(Extension):
            name = "swallow"

            def on_message(self, agent: object, msg: Message) -> None:
                raise StopIteration

        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("a")
        agent.use(SwallowExt())
        agent.register(sup)
        await sup.send(Message("x", "a", "hello"))
        await sup.run_once()
        assert received == []

    @pytest.mark.asyncio
    async def test_sync_on_load_and_on_unload(self) -> None:
        events: list[str] = []

        class TrackingExt(Extension):
            name = "tracking"

            def on_load(self, agent: object) -> None:
                events.append("load")

            def on_unload(self, agent: object) -> None:
                events.append("unload")

        agent = AsyncAgent("a")
        agent.use(TrackingExt())
        assert events == ["load"]
        agent.remove_extension("tracking")
        assert events == ["load", "unload"]

    @pytest.mark.asyncio
    async def test_replace_extension_calls_unload(self) -> None:
        events: list[str] = []

        class Ext1(Extension):
            name = "shared"

            def on_unload(self, agent: object) -> None:
                events.append("unload_1")

        class Ext2(Extension):
            name = "shared"

            def on_load(self, agent: object) -> None:
                events.append("load_2")

        agent = AsyncAgent("a")
        agent.use(Ext1())
        agent.use(Ext2())
        assert events == ["unload_1", "load_2"]

    @pytest.mark.asyncio
    async def test_remove_nonexistent_extension(self) -> None:
        agent = AsyncAgent("a")
        assert agent.remove_extension("nope") is False


# ---------------------------------------------------------------------------
# Async extension support on AsyncAgent
# ---------------------------------------------------------------------------


class TestAsyncAgentWithAsyncExtensions:
    """Async extensions (with ``async on_message``) work on :class:`AsyncAgent`."""

    @pytest.mark.asyncio
    async def test_async_extension_on_message(self) -> None:
        class AsyncUpperExt(Extension):
            name = "async_upper"

            async def on_message(self, agent: object, msg: Message) -> Message:
                return Message(msg.sender, msg.recipient, msg.content.upper())

        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("a")
        agent.use(AsyncUpperExt())
        agent.register(sup)
        await sup.send(Message("x", "a", "hello"))
        await sup.run_once()
        assert received == ["HELLO"]

    @pytest.mark.asyncio
    async def test_async_extension_swallows_message(self) -> None:
        class AsyncSwallow(Extension):
            name = "swallow"

            async def on_message(self, agent: object, msg: Message) -> None:
                raise StopIteration

        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("a")
        agent.use(AsyncSwallow())
        agent.register(sup)
        await sup.send(Message("x", "a", "hello"))
        await sup.run_once()
        assert received == []

    @pytest.mark.asyncio
    async def test_mixed_sync_and_async_extensions(self) -> None:
        """Both sync and async extensions chain correctly."""

        class SyncPrefix(Extension):
            name = "prefix"

            def on_message(self, agent: object, msg: Message) -> Message:
                return Message(msg.sender, msg.recipient, "PREFIX:" + msg.content)

        class AsyncSuffix(Extension):
            name = "suffix"

            async def on_message(self, agent: object, msg: Message) -> Message:
                return Message(msg.sender, msg.recipient, msg.content + ":SUFFIX")

        received: list[str] = []

        class MyAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = MyAgent("a")
        agent.use(SyncPrefix())
        agent.use(AsyncSuffix())
        agent.register(sup)
        await sup.send(Message("x", "a", "hello"))
        await sup.run_once()
        assert received == ["PREFIX:hello:SUFFIX"]


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAsyncIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_multi_agent_conversation(self) -> None:
        """Two async agents exchange messages via the supervisor."""
        log: list[str] = []

        class Ping(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                log.append(f"ping got: {msg.content}")
                if msg.content != "pong":
                    await self.send("ponger", "ping")

        class Pong(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                log.append(f"pong got: {msg.content}")
                await self.send("pinger", "pong")

        sup = AsyncSupervisor()
        pinger = Ping("pinger")
        ponger = Pong("ponger")
        pinger.register(sup)
        ponger.register(sup)

        await sup.send(Message("test", "pinger", "start"))
        await sup.run_once()  # pinger handles "start", sends "ping"
        assert "ping got: start" in log

        await sup.run_once()  # ponger handles "ping", sends "pong"
        assert "pong got: ping" in log

        await sup.run_once()  # pinger handles "pong", stops
        assert "ping got: pong" in log

    @pytest.mark.asyncio
    async def test_run_once_empty_returns_zero(self) -> None:
        sup = AsyncSupervisor()
        count = await sup.run_once()
        assert count == 0

    @pytest.mark.asyncio
    async def test_agent_count(self) -> None:
        sup = AsyncSupervisor()
        assert sup.agent_count() == 0
        agent = AsyncAgent("a")
        agent.register(sup)
        assert sup.agent_count() == 1
        agent.unregister()
        assert sup.agent_count() == 0

    @pytest.mark.asyncio
    async def test_multiple_agents_multiple_messages(self) -> None:
        results: dict[str, list[str]] = {"a": [], "b": []}

        class Collector(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                results[self.name].append(msg.content)

        sup = AsyncSupervisor()
        a = Collector("a")
        b = Collector("b")
        a.register(sup)
        b.register(sup)

        await sup.send(Message("x", "a", "m1"))
        await sup.send(Message("x", "b", "m2"))
        await sup.send(Message("x", "a", "m3"))
        await sup.run_once()
        assert sorted(results["a"]) == ["m1", "m3"]
        assert results["b"] == ["m2"]

    @pytest.mark.asyncio
    async def test_async_handler_with_await(self) -> None:
        """Ensure async handlers can genuinely ``await`` other coroutines."""
        received: list[str] = []

        class SlowAgent(AsyncAgent):
            async def handle_message(self, msg: Message) -> None:
                await asyncio.sleep(0.01)
                received.append(msg.content)

        sup = AsyncSupervisor()
        agent = SlowAgent("slow")
        agent.register(sup)
        await sup.send(Message("x", "slow", "delayed"))
        await sup.run_once()
        assert received == ["delayed"]

    @pytest.mark.asyncio
    async def test_sync_agent_unchanged(self) -> None:
        """Ensure the original sync Agent class is unaffected."""
        from supervisor.agent import Agent
        from supervisor._core import Supervisor

        received: list[str] = []

        class SyncAgent(Agent):
            def handle_message(self, msg: Message) -> None:
                received.append(msg.content)

        sup = Supervisor()
        agent = SyncAgent("sync")
        agent.register(sup)
        sup.send(Message("x", "sync", "hello"))
        sup.run_once()
        assert received == ["hello"]
