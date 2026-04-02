"""Anthropic LLM provider using httpx (no SDK dependency).

Implements the Anthropic Messages API for Claude models.
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

_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_API_VERSION = "2023-06-01"

_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-4-20250514": (3.0e-6, 15.0e-6),
    "claude-3-5-sonnet-20241022": (3.0e-6, 15.0e-6),
    "claude-3-5-haiku-20241022": (0.80e-6, 4.0e-6),
    "claude-3-opus-20240229": (15.0e-6, 75.0e-6),
}

_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError)

_retry_decorator = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


def _estimate_cost(model: str, usage: TokenUsage) -> float:
    prompt_rate, completion_rate = _PRICING.get(model, (0.0, 0.0))
    return usage.prompt_tokens * prompt_rate + usage.completion_tokens * completion_rate


class AnthropicProvider(BaseLLM):
    """Anthropic Claude provider.

    Parameters:
        model: Model name (default ``"claude-sonnet-4-20250514"``).
        api_key: API key.  Falls back to the ``ANTHROPIC_API_KEY`` env var.
        base_url: Override the API base URL.
        max_tokens: Maximum tokens to generate.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "Content-Type": "application/json",
        }

    def _messages_url(self) -> str:
        return f"{self.base_url}/messages"

    def _build_payload(
        self, prompt: str, stream: bool = False, **kwargs: object
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt:
            payload["system"] = system_prompt
        if stream:
            payload["stream"] = True
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        return payload

    def _parse_response(self, data: dict) -> LLMResponse:
        content_blocks = data.get("content", [])
        content = "".join(
            block.get("text", "") for block in content_blocks if block.get("type") == "text"
        )
        raw_usage = data.get("usage", {})
        usage = TokenUsage(
            prompt_tokens=raw_usage.get("input_tokens", 0),
            completion_tokens=raw_usage.get("output_tokens", 0),
            total_tokens=(
                raw_usage.get("input_tokens", 0)
                + raw_usage.get("output_tokens", 0)
            ),
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
        """Send a synchronous messages request."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._messages_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Yield text chunks from a streaming messages request."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream(
                "POST", self._messages_url(), headers=self._headers(), json=payload
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    event = json.loads(data_str)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text")
                        if text:
                            yield text

    # -- async ----------------------------------------------------------------

    @_retry_decorator
    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send an asynchronous messages request."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._messages_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Yield text chunks from an async streaming messages request."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                self._messages_url(),
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    event = json.loads(data_str)
                    if event.get("type") == "content_block_delta":
                        delta = event.get("delta", {})
                        text = delta.get("text")
                        if text:
                            yield text


__all__ = ["AnthropicProvider"]
