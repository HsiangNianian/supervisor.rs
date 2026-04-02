"""Tests for LLM providers with mocked HTTP responses."""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from supervisor.llm import (
    AnthropicProvider,
    AzureProvider,
    BaseLLM,
    BedrockProvider,
    LLMResponse,
    OllamaProvider,
    OpenAIProvider,
    TokenUsage,
    get_model,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _openai_chat_response(content: str = "Hello!", model: str = "gpt-4o-mini"):
    """Return a minimal OpenAI-style chat completion JSON dict."""
    return {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


def _anthropic_messages_response(
    content: str = "Hello!", model: str = "claude-sonnet-4-20250514"
):
    """Return a minimal Anthropic messages API JSON dict."""
    return {
        "id": "msg_abc123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "usage": {"input_tokens": 12, "output_tokens": 8},
    }


def _ollama_chat_response(content: str = "Hello!", model: str = "llama3"):
    """Return a minimal Ollama chat JSON dict."""
    return {
        "model": model,
        "message": {"role": "assistant", "content": content},
        "done": True,
        "prompt_eval_count": 20,
        "eval_count": 10,
    }


def _make_sync_response(data: dict, status_code: int = 200) -> httpx.Response:
    """Build a fake :class:`httpx.Response`."""
    return httpx.Response(
        status_code=status_code,
        json=data,
        request=httpx.Request("POST", "https://fake"),
    )


# ---------------------------------------------------------------------------
# TokenUsage / LLMResponse
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_defaults(self):
        u = TokenUsage()
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0

    def test_custom_values(self):
        u = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        assert u.total_tokens == 15


class TestLLMResponse:
    def test_creation(self):
        r = LLMResponse(content="hi", model="m")
        assert r.content == "hi"
        assert r.cost == 0.0
        assert r.usage.total_tokens == 0

    def test_with_usage(self):
        u = TokenUsage(prompt_tokens=3, completion_tokens=7, total_tokens=10)
        r = LLMResponse(content="ok", model="m", usage=u, cost=0.01)
        assert r.usage.prompt_tokens == 3
        assert r.cost == 0.01


# ---------------------------------------------------------------------------
# OpenAIProvider
# ---------------------------------------------------------------------------


class TestOpenAIProvider:
    def test_init_defaults(self):
        p = OpenAIProvider(api_key="sk-test")
        assert p.model == "gpt-4o-mini"
        assert p.api_key == "sk-test"

    def test_init_env_var(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        p = OpenAIProvider()
        assert p.api_key == "sk-env"

    @patch("httpx.Client")
    def test_invoke(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _make_sync_response(
            _openai_chat_response("Hi there!")
        )

        p = OpenAIProvider(api_key="sk-test")
        resp = p.invoke("Hello")

        assert isinstance(resp, LLMResponse)
        assert resp.content == "Hi there!"
        assert resp.model == "gpt-4o-mini"
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.total_tokens == 15
        mock_client.post.assert_called_once()

    @patch("httpx.Client")
    def test_stream(self, mock_client_cls):
        lines = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.iter_lines.return_value = iter(lines)

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_stream_resp
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        p = OpenAIProvider(api_key="sk-test")
        chunks = list(p.stream("Hello"))
        assert chunks == ["Hel", "lo"]

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _openai_chat_response("Async hi!")
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            p = OpenAIProvider(api_key="sk-test")
            resp = await p.ainvoke("Hello")

        assert resp.content == "Async hi!"

    @pytest.mark.asyncio
    async def test_astream(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"A"}}]}',
            'data: {"choices":[{"delta":{"content":"B"}}]}',
            "data: [DONE]",
        ]

        async def fake_aiter_lines():
            for line in lines:
                yield line

        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.aiter_lines = fake_aiter_lines

        # Build an async context manager that yields mock_stream_resp.
        stream_cm = MagicMock()
        stream_cm.__aenter__ = AsyncMock(return_value=mock_stream_resp)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        # Use MagicMock (not AsyncMock) so .stream() returns synchronously.
        mock_client = MagicMock()
        mock_client.stream.return_value = stream_cm

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            p = OpenAIProvider(api_key="sk-test")
            chunks = [chunk async for chunk in p.astream("Hello")]

        assert chunks == ["A", "B"]

    @patch("httpx.Client")
    def test_batch(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _make_sync_response(
            _openai_chat_response("Reply")
        )

        p = OpenAIProvider(api_key="sk-test")
        results = p.batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r.content == "Reply" for r in results)

    def test_repr(self):
        p = OpenAIProvider(model="gpt-4o", api_key="sk-test")
        assert "OpenAIProvider" in repr(p)
        assert "gpt-4o" in repr(p)


# ---------------------------------------------------------------------------
# AnthropicProvider
# ---------------------------------------------------------------------------


class TestAnthropicProvider:
    def test_init_defaults(self):
        p = AnthropicProvider(api_key="sk-ant-test")
        assert p.model == "claude-sonnet-4-20250514"

    @patch("httpx.Client")
    def test_invoke(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _make_sync_response(
            _anthropic_messages_response("Claude says hi")
        )

        p = AnthropicProvider(api_key="sk-ant-test")
        resp = p.invoke("Hello")

        assert resp.content == "Claude says hi"
        assert resp.usage.prompt_tokens == 12
        assert resp.usage.completion_tokens == 8
        assert resp.usage.total_tokens == 20

    @patch("httpx.Client")
    def test_stream(self, mock_client_cls):
        lines = [
            'event: content_block_delta',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"He"}}',
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"llo"}}',
            'data: {"type":"message_stop"}',
        ]
        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.iter_lines.return_value = iter(lines)

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_stream_resp
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        p = AnthropicProvider(api_key="sk-ant-test")
        chunks = list(p.stream("Hello"))
        assert chunks == ["He", "llo"]

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _anthropic_messages_response("Async Claude")
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            p = AnthropicProvider(api_key="sk-ant-test")
            resp = await p.ainvoke("Hello")

        assert resp.content == "Async Claude"


# ---------------------------------------------------------------------------
# OllamaProvider
# ---------------------------------------------------------------------------


class TestOllamaProvider:
    def test_init_defaults(self):
        p = OllamaProvider()
        assert p.model == "llama3"
        assert p.base_url == "http://localhost:11434"

    @patch("httpx.Client")
    def test_invoke(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _make_sync_response(
            _ollama_chat_response("Ollama says hi")
        )

        p = OllamaProvider()
        resp = p.invoke("Hello")

        assert resp.content == "Ollama says hi"
        assert resp.usage.prompt_tokens == 20
        assert resp.usage.completion_tokens == 10
        assert resp.cost == 0.0

    @patch("httpx.Client")
    def test_stream(self, mock_client_cls):
        lines = [
            json.dumps({"message": {"content": "Ol"}, "done": False}),
            json.dumps({"message": {"content": "lama"}, "done": True}),
        ]
        mock_stream_resp = MagicMock()
        mock_stream_resp.raise_for_status = MagicMock()
        mock_stream_resp.iter_lines.return_value = iter(lines)

        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.stream.return_value.__enter__ = MagicMock(
            return_value=mock_stream_resp
        )
        mock_client.stream.return_value.__exit__ = MagicMock(return_value=False)

        p = OllamaProvider()
        chunks = list(p.stream("Hello"))
        assert chunks == ["Ol", "lama"]

    @pytest.mark.asyncio
    async def test_ainvoke(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _ollama_chat_response("Async Ollama")
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_resp

        with patch("httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            p = OllamaProvider()
            resp = await p.ainvoke("Hello")

        assert resp.content == "Async Ollama"


# ---------------------------------------------------------------------------
# AzureProvider
# ---------------------------------------------------------------------------


class TestAzureProvider:
    def test_init_defaults(self):
        p = AzureProvider(
            api_key="az-key",
            endpoint="https://myresource.openai.azure.com",
        )
        assert p.model == "gpt-4o"
        assert "myresource" in p.endpoint

    def test_init_env_var(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "az-env-key")
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://env.openai.azure.com")
        p = AzureProvider()
        assert p.api_key == "az-env-key"
        assert "env" in p.endpoint

    @patch("httpx.Client")
    def test_invoke(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
        mock_client.post.return_value = _make_sync_response(
            _openai_chat_response("Azure says hi", model="gpt-4o")
        )

        p = AzureProvider(
            api_key="az-key",
            endpoint="https://myresource.openai.azure.com",
        )
        resp = p.invoke("Hello")

        assert resp.content == "Azure says hi"


# ---------------------------------------------------------------------------
# BedrockProvider
# ---------------------------------------------------------------------------


class TestBedrockProvider:
    def test_requires_boto3(self):
        with pytest.raises(ImportError, match="boto3"):
            BedrockProvider()

    def test_invoke_stub(self):
        """Verify the stub raises NotImplementedError even if boto3 exists."""
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            p = BedrockProvider()
            with pytest.raises(NotImplementedError):
                p.invoke("test")

    def test_stream_stub(self):
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            p = BedrockProvider()
            with pytest.raises(NotImplementedError):
                list(p.stream("test"))


# ---------------------------------------------------------------------------
# get_model factory
# ---------------------------------------------------------------------------


class TestGetModel:
    def test_openai_prefix(self):
        m = get_model("openai:gpt-4o", api_key="sk-test")
        assert isinstance(m, OpenAIProvider)
        assert m.model == "gpt-4o"

    def test_anthropic_prefix(self):
        m = get_model("anthropic:claude-sonnet-4-20250514", api_key="k")
        assert isinstance(m, AnthropicProvider)
        assert m.model == "claude-sonnet-4-20250514"

    def test_ollama_prefix(self):
        m = get_model("ollama:mistral")
        assert isinstance(m, OllamaProvider)
        assert m.model == "mistral"

    def test_bare_name_with_model_kwarg(self):
        m = get_model("openai", model="gpt-4o-mini", api_key="sk-test")
        assert isinstance(m, OpenAIProvider)
        assert m.model == "gpt-4o-mini"

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_model("foobar:model-x")

    def test_case_insensitive(self):
        m = get_model("OpenAI:gpt-4o", api_key="sk-test")
        assert isinstance(m, OpenAIProvider)

    def test_azure(self):
        m = get_model(
            "azure:gpt-4o",
            api_key="k",
            endpoint="https://x.openai.azure.com",
        )
        assert isinstance(m, AzureProvider)
        assert m.model == "gpt-4o"


# ---------------------------------------------------------------------------
# BaseLLM ABC
# ---------------------------------------------------------------------------


class TestBaseLLM:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseLLM("model")  # type: ignore[abstract]
