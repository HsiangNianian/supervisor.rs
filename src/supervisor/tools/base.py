"""Base tool abstract class for the supervisor tool ecosystem."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for all tools.

    Attributes:
        name: The unique name of the tool.
        description: A human-readable description of what the tool does.
        parameters: JSON Schema describing the tool's parameters.
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool synchronously.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            Tool-specific result.
        """

    async def aexecute(self, **kwargs: Any) -> Any:
        """Execute the tool asynchronously.

        Default implementation runs execute() in a thread pool.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            Tool-specific result.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.execute(**kwargs))

    def to_openai_spec(self) -> dict:
        """Export the tool as an OpenAI function calling specification.

        Returns:
            Dict in OpenAI function calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_schema(self) -> dict:
        """Export the tool as a JSON schema.

        Returns:
            Dict containing the tool's JSON schema representation.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
