"""Azure OpenAI LLM provider using httpx (no SDK dependency).

Implements the Azure OpenAI chat completions endpoint which uses a
deployment-based URL scheme and ``api-key`` header authentication.
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

_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError)

_retry_decorator = retry(
    retry=retry_if_exception_type(_RETRYABLE),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class AzureProvider(BaseLLM):
    """Azure OpenAI chat-completions provider.

    Parameters:
        model: The deployment name.
        api_key: API key.  Falls back to ``AZURE_OPENAI_API_KEY``.
        endpoint: Azure OpenAI resource endpoint
            (e.g. ``https://my-resource.openai.azure.com``).
            Falls back to ``AZURE_OPENAI_ENDPOINT``.
        api_version: API version string.
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        *,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str = "2024-02-01",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(model)
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self.endpoint = (
            endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        ).rstrip("/")
        self.api_version = api_version
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        return (
            f"{self.endpoint}/openai/deployments/{self.model}"
            f"/chat/completions?api-version={self.api_version}"
        )

    def _build_payload(
        self, prompt: str, stream: bool = False, **kwargs: object
    ) -> dict[str, object]:
        messages: list[dict[str, str]] = []
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
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
            cost=0.0,
        )

    # -- sync -----------------------------------------------------------------

    @_retry_decorator
    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send a synchronous chat completion request to Azure OpenAI."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self._chat_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Yield text chunks from a streaming Azure OpenAI request."""
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
        """Send an asynchronous chat completion request to Azure OpenAI."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                self._chat_url(), headers=self._headers(), json=payload
            )
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Yield text chunks from an async streaming Azure OpenAI request."""
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


__all__ = ["AzureProvider"]
