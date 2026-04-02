"""Tests for LLMAgent with mocked LLM provider."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from supervisor._core import Message, Supervisor
from supervisor.llm.base import BaseLLM, LLMResponse, TokenUsage
from supervisor.llm_agent import LLMAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLLM(BaseLLM):
    """A minimal concrete LLM for testing that returns canned responses."""

    def __init__(self, response: str = "I am an LLM.") -> None:
        super().__init__("fake-model")
        self._response = response
        self.call_count = 0
        self.last_kwargs: dict = {}

    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        self.call_count += 1
        self.last_kwargs = dict(kwargs)
        return LLMResponse(
            content=self._response,
            model=self.model,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=3, total_tokens=8),
            cost=0.001,
        )

    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        return self.invoke(prompt, **kwargs)

    def stream(self, prompt: str, **kwargs: object):
        yield self._response

    async def astream(self, prompt: str, **kwargs: object):
        yield self._response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMAgentCreation:
    def test_basic_creation(self):
        llm = FakeLLM()
        agent = LLMAgent("assistant", llm=llm)
        assert agent.name == "assistant"
        assert agent.llm is llm
        assert agent.system_prompt == ""
        assert agent.max_history == 50
        assert agent.history == []

    def test_custom_params(self):
        llm = FakeLLM()
        agent = LLMAgent(
            "bot",
            llm=llm,
            system_prompt="You are helpful.",
            max_history=10,
        )
        assert agent.system_prompt == "You are helpful."
        assert agent.max_history == 10

    def test_repr(self):
        llm = FakeLLM()
        agent = LLMAgent("bot", llm=llm)
        r = repr(agent)
        assert "LLMAgent" in r
        assert "bot" in r


class TestLLMAgentHandleMessage:
    def test_handle_message_calls_llm(self):
        llm = FakeLLM("LLM reply")
        agent = LLMAgent("assistant", llm=llm)

        # Register with supervisor so send() works.
        sup = Supervisor()
        agent.register(sup)
        # Register a dummy receiver for the reply.
        received = []
        sup.register("user", lambda msg: received.append(msg.content))

        msg = Message("user", "assistant", "Hello!")
        agent.handle_message(msg)

        assert llm.call_count == 1
        assert agent.last_response is not None
        assert agent.last_response.content == "LLM reply"

    def test_handle_message_updates_history(self):
        llm = FakeLLM("Response 1")
        agent = LLMAgent("assistant", llm=llm)
        sup = Supervisor()
        agent.register(sup)

        agent.handle_message(Message("user", "assistant", "Q1"))

        assert len(agent.history) == 2
        assert agent.history[0] == {"role": "user", "content": "Q1"}
        assert agent.history[1] == {"role": "assistant", "content": "Response 1"}

    def test_handle_message_sends_reply(self):
        llm = FakeLLM("The answer is 4")
        agent = LLMAgent("assistant", llm=llm)
        sup = Supervisor()
        agent.register(sup)

        received = []
        sup.register("user", lambda msg: received.append(msg.content))

        agent.handle_message(Message("user", "assistant", "What is 2+2?"))

        # The reply is queued; deliver it.
        sup.run_once()
        assert received == ["The answer is 4"]

    def test_handle_message_no_error_if_sender_unregistered(self):
        """If sender is not registered, the reply send should be silently ignored."""
        llm = FakeLLM("Reply")
        agent = LLMAgent("assistant", llm=llm)
        sup = Supervisor()
        agent.register(sup)
        # "ghost" is not registered — should not raise.
        agent.handle_message(Message("ghost", "assistant", "Hi"))
        assert llm.call_count == 1

    def test_messages_include_system_prompt(self):
        llm = FakeLLM("OK")
        agent = LLMAgent(
            "assistant", llm=llm, system_prompt="Be concise."
        )
        sup = Supervisor()
        agent.register(sup)

        agent.handle_message(Message("user", "assistant", "Hello"))

        # Check that the messages kwarg includes the system prompt.
        messages = llm.last_kwargs.get("messages", [])
        assert messages[0] == {"role": "system", "content": "Be concise."}

    def test_messages_include_history(self):
        llm = FakeLLM("Reply")
        agent = LLMAgent("assistant", llm=llm)
        sup = Supervisor()
        agent.register(sup)

        agent.handle_message(Message("user", "assistant", "First"))
        agent.handle_message(Message("user", "assistant", "Second"))

        messages = llm.last_kwargs.get("messages", [])
        # Should contain: history(user:First, assistant:Reply) + user:Second
        user_contents = [m["content"] for m in messages if m["role"] == "user"]
        assert "First" in user_contents
        assert "Second" in user_contents


class TestLLMAgentHistory:
    def test_clear_history(self):
        llm = FakeLLM("Reply")
        agent = LLMAgent("assistant", llm=llm)
        sup = Supervisor()
        agent.register(sup)

        agent.handle_message(Message("user", "assistant", "Q1"))
        assert len(agent.history) == 2

        agent.clear_history()
        assert agent.history == []

    def test_max_history_trims(self):
        llm = FakeLLM("R")
        agent = LLMAgent("assistant", llm=llm, max_history=2)
        sup = Supervisor()
        agent.register(sup)

        # Send 3 messages (max_history=2 → keep 4 messages = 2 turns).
        for i in range(3):
            agent.handle_message(Message("user", "assistant", f"Q{i}"))

        # Should only keep the last 2 turns (4 messages).
        assert len(agent.history) == 4
        # The oldest turn (Q0) should be gone.
        contents = [m["content"] for m in agent.history]
        assert "Q0" not in contents
        assert "Q1" in contents
        assert "Q2" in contents

    def test_max_history_zero_disables(self):
        llm = FakeLLM("R")
        agent = LLMAgent("assistant", llm=llm, max_history=0)
        sup = Supervisor()
        agent.register(sup)

        agent.handle_message(Message("user", "assistant", "Q1"))
        assert agent.history == []

    def test_last_response_property(self):
        llm = FakeLLM("First")
        agent = LLMAgent("assistant", llm=llm)
        assert agent.last_response is None

        sup = Supervisor()
        agent.register(sup)
        agent.handle_message(Message("user", "assistant", "Q"))
        assert agent.last_response is not None
        assert agent.last_response.content == "First"


class TestLLMAgentWithSupervisor:
    def test_full_roundtrip(self):
        """Verify LLM agent processes message and records history.

        Note: The Rust-backed Supervisor does not allow re-entrant sends
        during ``run_once`` dispatch, so the reply to the sender is
        silently dropped.  We verify the LLM was called and history updated.
        """
        llm = FakeLLM("42")
        agent = LLMAgent("calculator", llm=llm)

        sup = Supervisor()
        agent.register(sup)
        sup.register("user", lambda msg: None)

        sup.send(Message("user", "calculator", "What is 6*7?"))
        sup.run_once()

        assert llm.call_count == 1
        assert agent.last_response is not None
        assert agent.last_response.content == "42"
        assert len(agent.history) == 2

    def test_multi_turn_conversation(self):
        """Multiple turns should accumulate history."""
        call_num = 0

        class CountingLLM(FakeLLM):
            def invoke(self, prompt, **kwargs):
                nonlocal call_num
                call_num += 1
                self._response = f"Answer {call_num}"
                return super().invoke(prompt, **kwargs)

        llm = CountingLLM()
        agent = LLMAgent("bot", llm=llm)
        sup = Supervisor()
        agent.register(sup)

        for i in range(5):
            agent.handle_message(Message("user", "bot", f"Turn {i}"))

        assert len(agent.history) == 10  # 5 turns × 2 messages each
        assert agent.last_response.content == "Answer 5"

    def test_llm_agent_is_agent_subclass(self):
        from supervisor.agent import Agent

        llm = FakeLLM()
        agent = LLMAgent("test", llm=llm)
        assert isinstance(agent, Agent)
