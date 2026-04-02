"""Base LLM abstraction layer.

Defines :class:`BaseLLM`, the abstract interface that all LLM providers
implement, together with the shared response types :class:`LLMResponse`
and :class:`TokenUsage`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage statistics for a single LLM call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class LLMResponse(BaseModel):
    """Response from an LLM provider."""

    content: str
    model: str
    usage: TokenUsage = Field(default_factory=TokenUsage)
    cost: float = 0.0


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement :meth:`invoke`, :meth:`ainvoke`,
    :meth:`stream`, and :meth:`astream`.  A default :meth:`batch`
    implementation is provided that calls :meth:`invoke` in a loop.

    Parameters:
        model: The model identifier (e.g. ``"gpt-4o"``).
    """

    def __init__(self, model: str) -> None:
        self.model = model

    @abstractmethod
    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Synchronous single-prompt call."""

    @abstractmethod
    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Asynchronous single-prompt call."""

    @abstractmethod
    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Synchronous streaming call yielding text chunks."""

    @abstractmethod
    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Asynchronous streaming call yielding text chunks."""
        # Yield to make this an async generator in subclasses.
        yield  # type: ignore[misc]  # pragma: no cover

    def batch(self, prompts: list[str], **kwargs: object) -> list[LLMResponse]:
        """Process multiple prompts synchronously.

        The default implementation calls :meth:`invoke` for each prompt.
        Subclasses may override for concurrent/parallel execution.
        """
        return [self.invoke(prompt, **kwargs) for prompt in prompts]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


__all__ = ["BaseLLM", "LLMResponse", "TokenUsage"]
