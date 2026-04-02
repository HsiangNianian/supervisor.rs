"""LLM abstraction layer for the supervisor agent framework.

Provides a unified interface to multiple LLM providers (OpenAI, Anthropic,
Ollama, Azure OpenAI, AWS Bedrock) and a :func:`get_model` factory for
instantiating providers by name.
"""

from __future__ import annotations

from supervisor.llm.anthropic_provider import AnthropicProvider
from supervisor.llm.azure_provider import AzureProvider
from supervisor.llm.base import BaseLLM, LLMResponse, TokenUsage
from supervisor.llm.bedrock_provider import BedrockProvider
from supervisor.llm.ollama_provider import OllamaProvider
from supervisor.llm.openai_provider import OpenAIProvider

_PROVIDER_ALIASES: dict[str, type[BaseLLM]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "azure": AzureProvider,
    "bedrock": BedrockProvider,
}


def get_model(name: str, **kwargs: object) -> BaseLLM:
    """Instantiate an LLM provider by short name or model string.

    The *name* is matched against known provider prefixes (e.g.
    ``"openai:gpt-4o"`` or ``"anthropic:claude-sonnet-4-20250514"``).
    If *name* contains no prefix, the bare string is treated as a provider
    key.

    Examples::

        get_model("openai:gpt-4o")
        get_model("anthropic:claude-sonnet-4-20250514", api_key="sk-...")
        get_model("ollama:llama3")
        get_model("openai", model="gpt-4o-mini")

    Raises:
        ValueError: If the provider name is not recognised.
    """
    if ":" in name:
        provider_name, model_name = name.split(":", 1)
        kwargs.setdefault("model", model_name)
    else:
        provider_name = name

    provider_cls = _PROVIDER_ALIASES.get(provider_name.lower())
    if provider_cls is None:
        available = ", ".join(sorted(_PROVIDER_ALIASES))
        raise ValueError(
            f"Unknown provider {provider_name!r}. "
            f"Available providers: {available}"
        )

    model_arg = kwargs.pop("model", None)
    if model_arg is not None:
        return provider_cls(model=str(model_arg), **kwargs)  # type: ignore[arg-type]
    return provider_cls(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "AnthropicProvider",
    "AzureProvider",
    "BaseLLM",
    "BedrockProvider",
    "LLMResponse",
    "OllamaProvider",
    "OpenAIProvider",
    "TokenUsage",
    "get_model",
]
