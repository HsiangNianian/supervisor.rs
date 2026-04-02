"""LLM-powered agent that integrates with the supervisor framework.

Provides :class:`LLMAgent`, a subclass of :class:`~supervisor.agent.Agent`
that delegates message handling to an LLM provider with conversation history
management.
"""

from __future__ import annotations

from typing import Optional

from supervisor._core import Message
from supervisor.agent import Agent
from supervisor.llm.base import BaseLLM, LLMResponse


class LLMAgent(Agent):
    """An agent that processes messages using an LLM.

    Maintains a sliding-window conversation history and sends the full
    context to the LLM on each incoming message.  The LLM response is
    sent back to the original sender.

    Parameters:
        name: Unique agent name.
        llm: The LLM provider to use for inference.
        system_prompt: Optional system prompt prepended to every request.
        max_history: Maximum number of conversation turns (user + assistant
            pairs) to keep.  ``0`` disables history.

    Example::

        from supervisor import Supervisor, Message
        from supervisor.llm import OpenAIProvider
        from supervisor.llm_agent import LLMAgent

        llm = OpenAIProvider(api_key="sk-...")
        agent = LLMAgent("assistant", llm=llm, system_prompt="You are helpful.")

        sup = Supervisor()
        agent.register(sup)
        sup.send(Message("user", "assistant", "What is 2+2?"))
        sup.run_once()
    """

    def __init__(
        self,
        name: str,
        *,
        llm: BaseLLM,
        system_prompt: str = "",
        max_history: int = 50,
    ) -> None:
        super().__init__(name)
        self.llm = llm
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.history: list[dict[str, str]] = []
        self._last_response: Optional[LLMResponse] = None

    def _build_messages(self, user_content: str) -> list[dict[str, str]]:
        """Build the full message list including system prompt and history."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_content})
        return messages

    def _trim_history(self) -> None:
        """Trim history to ``max_history`` most recent turns."""
        if self.max_history <= 0:
            self.history.clear()
            return
        # Each turn is a user + assistant message pair.
        max_messages = self.max_history * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def handle_message(self, msg: Message) -> None:
        """Process an incoming message by sending it to the LLM.

        The LLM response is stored in history and sent back to the
        original sender via the supervisor.
        """
        messages = self._build_messages(msg.content)
        response = self.llm.invoke(
            msg.content,
            messages=messages,
        )
        self._last_response = response

        # Record the exchange in history.
        self.history.append({"role": "user", "content": msg.content})
        self.history.append({"role": "assistant", "content": response.content})
        self._trim_history()

        # Reply to the sender.
        if self.supervisor is not None:
            try:
                self.send(msg.sender, response.content)
            except KeyError:
                # Sender may not be a registered agent (e.g. "user").
                pass

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history.clear()

    @property
    def last_response(self) -> Optional[LLMResponse]:
        """The most recent LLM response, or ``None``."""
        return self._last_response

    def __repr__(self) -> str:
        return (
            f"LLMAgent(name={self.name!r}, llm={self.llm!r}, "
            f"history_len={len(self.history)})"
        )


__all__ = ["LLMAgent"]
