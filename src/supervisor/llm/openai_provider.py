"""OpenAI LLM provider using httpx (no SDK dependency).

Supports both synchronous and asynchronous calls, streaming, and
automatic retry with exponential back-off via *tenacity*.
"""

from __future__ import annotations

import json
import os
from typing import AsyncIterator, Iterator

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from supervisor.llm.base import BaseLLM, LLMResponse, TokenUsage

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Rough per-token pricing (USD) for common models — used only for estimates.
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50e-6, 10.0e-6),
    "gpt-4o-mini": (0.15e-6, 0.60e-6),
    "gpt-4-turbo": (10.0e-6, 30.0e-6),
    "gpt-4": (30.0e-6, 60.0e-6),
    "gpt-3.5-turbo": (0.50e-6, 1.50e-6),
}

_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError)

_retry_decorator = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


def _estimate_cost(model: str, usage: TokenUsage) -> float:
    """Return estimated cost in USD for *usage* on *model*."""
    prompt_rate, completion_rate = _PRICING.get(model, (0.0, 0.0))
    return usage.prompt_tokens * prompt_rate + usage.completion_tokens * completion_rate


def _build_messages(
    prompt: str, system_prompt: str | None = None
) -> list[dict[str, str]]:
    """Build the ``messages`` payload for the chat completions API."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


class OpenAIProvider(BaseLLM):
    """OpenAI chat-completions provider.

    Parameters:
        model: Model name (default ``"gpt-4o-mini"``).
        api_key: API key.  Falls back to the ``OPENAI_API_KEY`` env var.
        base_url: Override the API base URL.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _build_payload(
        self, prompt: str, stream: bool = False, **kwargs: object
    ) -> dict[str, object]:
        system_prompt = kwargs.pop("system_prompt", None)
        messages = kwargs.pop("messages", None) or _build_messages(
            prompt, system_prompt  # type: ignore[arg-type]
        )
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        return payload

    def _parse_response(self, data: dict) -> LLMResponse:
        choice = data["choices"][0]
        content = choice["message"]["content"] or ""
        raw_usage = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            cost=_estimate_cost(self.model, usage),
        )

    # -- sync -----------------------------------------------------------------

    @_retry_decorator
    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send a synchronous chat completion request."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._chat_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Yield text chunks from a streaming chat completion."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", self._chat_url(), headers=self._headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text

    # -- async ----------------------------------------------------------------

    @_retry_decorator
    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send an asynchronous chat completion request."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._chat_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Yield text chunks from an async streaming chat completion."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self._chat_url(), headers=self._headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text


__all__ = ["OpenAIProvider"]
