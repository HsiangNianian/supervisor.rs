"""AWS Bedrock LLM provider (stub).

Provides a placeholder implementation for Amazon Bedrock.  Full support
requires the ``boto3`` library and AWS credential configuration.  When
``boto3`` is not installed the provider raises :class:`ImportError` at
instantiation time.
"""

from __future__ import annotations

from typing import AsyncIterator, Iterator

from supervisor.llm.base import BaseLLM, LLMResponse, TokenUsage


class BedrockProvider(BaseLLM):
    """AWS Bedrock provider (stub).

    Parameters:
        model: Bedrock model ID (default ``"anthropic.claude-3-sonnet-20240229-v1:0"``).
        region: AWS region name.

    Raises:
        ImportError: If ``boto3`` is not installed.
    """

    def __init__(
        self,
        model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
        *,
        region: str = "us-east-1",
    ) -> None:
        super().__init__(model)
        self.region = region
        try:
            import boto3  # noqa: F401

            self._boto3 = boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for BedrockProvider. "
                "Install it with: pip install boto3"
            ) from None

    def _get_client(self):  # type: ignore[no-untyped-def]
        """Return a Bedrock runtime client."""
        return self._boto3.client(
            "bedrock-runtime", region_name=self.region
        )

    def invoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Invoke the Bedrock model synchronously.

        This is a stub — real implementation would use ``boto3``'s
        ``invoke_model`` API.
        """
        raise NotImplementedError(
            "BedrockProvider.invoke is a stub. "
            "Full Bedrock integration is planned for a future release."
        )

    async def ainvoke(self, prompt: str, **kwargs: object) -> LLMResponse:
        """Invoke the Bedrock model asynchronously (stub)."""
        raise NotImplementedError(
            "BedrockProvider.ainvoke is a stub. "
            "Full Bedrock integration is planned for a future release."
        )

    def stream(self, prompt: str, **kwargs: object) -> Iterator[str]:
        """Stream from the Bedrock model (stub)."""
        raise NotImplementedError(
            "BedrockProvider.stream is a stub. "
            "Full Bedrock integration is planned for a future release."
        )

    async def astream(self, prompt: str, **kwargs: object) -> AsyncIterator[str]:
        """Async stream from the Bedrock model (stub)."""
        raise NotImplementedError(
            "BedrockProvider.astream is a stub. "
            "Full Bedrock integration is planned for a future release."
        )
        yield  # type: ignore[misc]  # pragma: no cover


__all__ = ["BedrockProvider"]
