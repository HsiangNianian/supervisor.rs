"""Ollama LLM provider using httpx (no SDK dependency).

Connects to a local Ollama instance for inference with open-source models.
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterator

import httpx

from supervisor.llm.base import BaseLLM, LLMResponse, TokenUsage

_DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(BaseLLM):
    """Ollama local inference provider.

    Parameters:
        model: Model name (default ``"llama3"``).
        base_url: Ollama server URL (default ``http://localhost:11434``).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3",
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = 120.0,
    ) -> None:
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def _build_payload(
        self, prompt: str, stream: bool = False, **kwargs: object
    ) -> dict[str, object]:
        messages: list[dict[str, str]] = []
        system_prompt = kwargs.pop("system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": str(system_prompt)})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, object] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        options: dict[str, object] = {}
        if "temperature" in kwargs:
            options["temperature"] = kwargs["temperature"]
        if options:
            payload["options"] = options
        return payload

    def _parse_response(self, data: dict) -> LLMResponse:
        message = data.get("message", {})
        content = message.get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0) or 0
        completion_tokens = data.get("eval_count", 0) or 0
        usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            usage=usage,
            cost=0.0,
        )

    # -- sync -----------------------------------------------------------------

    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send a synchronous chat request to Ollama."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._chat_url(), json=payload)
            resp.raise_for_status()
            return self._parse_response(resp.json())

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Yield text chunks from a streaming Ollama chat request."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self._chat_url(), json=payload) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("message", {}).get("content")
                    if text:
                        yield text

    # -- async ----------------------------------------------------------------

    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Send an asynchronous chat request to Ollama."""
        payload = self._build_payload(prompt, stream=False, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self._chat_url(), json=payload)
            resp.raise_for_status()
            return self._parse_response(resp.json())

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Yield text chunks from an async streaming Ollama chat request."""
        payload = self._build_payload(prompt, stream=True, **kwargs)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", self._chat_url(), json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    text = chunk.get("message", {}).get("content")
                    if text:
                        yield text


__all__ = ["OllamaProvider"]
