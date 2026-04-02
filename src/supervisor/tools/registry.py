"""Tool registry for discovery and management."""

from __future__ import annotations

from typing import Any

from supervisor.tools.base import BaseTool


class ToolRegistry:
    """Central registry for tool discovery and management.

    Provides registration, lookup, search, and bulk export of tools.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool.

        Args:
            tool: The tool instance to register.
        """
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get a tool by name.

        Args:
            name: Tool name.

        Returns:
            The tool instance.

        Raises:
            KeyError: If no tool with that name is registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return self._tools[name]

    def list_tools(self) -> list[BaseTool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def search(self, query: str) -> list[BaseTool]:
        """Search tools by keyword in name or description.

        Args:
            query: Search keyword.

        Returns:
            List of matching tools.
        """
        q = query.lower()
        return [
            t
            for t in self._tools.values()
            if q in t.name.lower() or q in t.description.lower()
        ]

    def get_openai_tools(self) -> list[dict]:
        """Export all tools in OpenAI function calling format."""
        return [t.to_openai_spec() for t in self._tools.values()]

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a tool by name.

        Args:
            name: Tool name.
            **kwargs: Tool parameters.

        Returns:
            Tool execution result.
        """
        return self.get(name).execute(**kwargs)
