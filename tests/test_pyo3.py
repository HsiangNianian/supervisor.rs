"""Tests for the supervisor agent framework."""

import pytest

from supervisor._core import Message, Supervisor


class TestMessage:
    def test_creation(self):
        msg = Message("alice", "bob", "hello")
        assert msg.sender == "alice"
        assert msg.recipient == "bob"
        assert msg.content == "hello"

    def test_repr(self):
        msg = Message("a", "b", "c")
        r = repr(msg)
        assert "sender" in r
        assert "recipient" in r
        assert "content" in r

    def test_str(self):
        msg = Message("sender", "recipient", "payload")
        s = str(msg)
        assert "sender" in s
        assert "recipient" in s
        assert "payload" in s

    def test_mutable_fields(self):
        msg = Message("x", "y", "z")
        msg.sender = "new_sender"
        msg.content = "new_content"
        assert msg.sender == "new_sender"
        assert msg.content == "new_content"


class TestSupervisor:
    def test_empty_supervisor(self):
        sup = Supervisor()
        assert sup.agent_count() == 0
        assert sup.agent_names() == []

    def test_register_agent(self):
        sup = Supervisor()
        sup.register("agent1", lambda msg: None)
        assert "agent1" in sup.agent_names()
        assert sup.agent_count() == 1

    def test_register_multiple_agents(self):
        sup = Supervisor()
        sup.register("agent1", lambda msg: None)
        sup.register("agent2", lambda msg: None)
        assert set(sup.agent_names()) == {"agent1", "agent2"}
        assert sup.agent_count() == 2

    def test_unregister_existing_agent(self):
        sup = Supervisor()
        sup.register("agent1", lambda msg: None)
        result = sup.unregister("agent1")
        assert result is True
        assert "agent1" not in sup.agent_names()

    def test_unregister_nonexistent_agent(self):
        sup = Supervisor()
        result = sup.unregister("ghost")
        assert result is False

    def test_send_to_registered_agent(self):
        sup = Supervisor()
        sup.register("target", lambda msg: None)
        # Should not raise
        sup.send(Message("system", "target", "hello"))
        assert sup.pending_count("target") == 1

    def test_send_to_unknown_agent_raises(self):
        sup = Supervisor()
        with pytest.raises(KeyError):
            sup.send(Message("system", "nobody", "hello"))

    def test_pending_count_known_agent(self):
        sup = Supervisor()
        sup.register("agent", lambda msg: None)
        assert sup.pending_count("agent") == 0
        sup.send(Message("x", "agent", "msg1"))
        sup.send(Message("x", "agent", "msg2"))
        assert sup.pending_count("agent") == 2

    def test_pending_count_unknown_agent(self):
        sup = Supervisor()
        assert sup.pending_count("ghost") is None

    def test_run_once_delivers_messages(self):
        sup = Supervisor()
        received: list[str] = []
        sup.register("collector", lambda msg: received.append(msg.content))
        sup.send(Message("src", "collector", "first"))
        sup.send(Message("src", "collector", "second"))
        count = sup.run_once()
        assert count == 2
        assert received == ["first", "second"]

    def test_run_once_clears_queue(self):
        sup = Supervisor()
        sup.register("agent", lambda msg: None)
        sup.send(Message("x", "agent", "msg"))
        sup.run_once()
        assert sup.pending_count("agent") == 0

    def test_run_once_returns_zero_when_no_messages(self):
        sup = Supervisor()
        sup.register("agent", lambda msg: None)
        count = sup.run_once()
        assert count == 0

    def test_fault_tolerant_run_once(self):
        """Supervisor continues processing after an agent raises an exception."""
        sup = Supervisor()
        good_received: list[str] = []

        def bad_handler(msg):
            raise RuntimeError("simulated agent failure")

        def good_handler(msg):
            good_received.append(msg.content)

        sup.register("bad", bad_handler)
        sup.register("good", good_handler)
        sup.send(Message("x", "bad", "crash me"))
        sup.send(Message("x", "good", "survive"))

        # Should not raise even though "bad" agent fails; only the successful
        # delivery to "good" is counted.
        count = sup.run_once()
        assert count == 1
        # "good" agent still processed its message
        assert good_received == ["survive"]

    def test_message_content_received_correctly(self):
        sup = Supervisor()
        messages_received: list[Message] = []
        sup.register("inbox", lambda msg: messages_received.append(msg))
        sent = Message("alice", "inbox", "greetings")
        sup.send(sent)
        sup.run_once()
        assert len(messages_received) == 1
        assert messages_received[0].sender == "alice"
        assert messages_received[0].content == "greetings"

    def test_multiple_agents_independent_queues(self):
        sup = Supervisor()
        results: dict[str, list[str]] = {"a": [], "b": []}
        sup.register("a", lambda msg: results["a"].append(msg.content))
        sup.register("b", lambda msg: results["b"].append(msg.content))
        sup.send(Message("x", "a", "for_a"))
        sup.send(Message("x", "b", "for_b"))
        sup.run_once()
        assert results["a"] == ["for_a"]
        assert results["b"] == ["for_b"]
