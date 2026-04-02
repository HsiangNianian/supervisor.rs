"""Third-party tool integration adapters."""

from __future__ import annotations

from typing import Any

from supervisor.tools.base import BaseTool
from supervisor.tools.registry import ToolRegistry


class LangChainToolAdapter(BaseTool):
    """Adapter to wrap a LangChain-style tool as a BaseTool.

    LangChain tools typically have ``name``, ``description``, and ``_run`` attrs.

    Args:
        lc_tool: A LangChain tool object.
    """

    def __init__(self, lc_tool: Any) -> None:
        self._lc_tool = lc_tool
        self.name = getattr(lc_tool, "name", "langchain_tool")
        self.description = getattr(lc_tool, "description", "")

    def execute(self, **kwargs: Any) -> Any:
        """Execute the wrapped LangChain tool.

        Args:
            **kwargs: Passed to the tool's ``_run`` method.

        Returns:
            Tool result.
        """
        if hasattr(self._lc_tool, "_run"):
            return self._lc_tool._run(**kwargs)
        if callable(self._lc_tool):
            return self._lc_tool(**kwargs)
        raise RuntimeError("LangChain tool has no _run method")

    @classmethod
    def from_langchain(cls, tool: Any) -> LangChainToolAdapter:
        """Create an adapter from a LangChain tool.

        Args:
            tool: LangChain tool instance.

        Returns:
            Wrapped adapter.
        """
        return cls(tool)


class OpenAIFunctionAdapter:
    """Convert tools to/from OpenAI function calling format."""

    @staticmethod
    def to_openai_functions(tools: list[BaseTool]) -> list[dict]:
        """Convert tools to OpenAI function calling spec.

        Args:
            tools: List of BaseTool instances.

        Returns:
            List of OpenAI function definitions.
        """
        return [t.to_openai_spec() for t in tools]

    @staticmethod
    def call_from_openai_response(
        response: dict, registry: ToolRegistry
    ) -> Any:
        """Execute a tool from an OpenAI function call response.

        Args:
            response: Dict with 'name' and 'arguments' keys.
            registry: ToolRegistry to look up the tool.

        Returns:
            Tool execution result.
        """
        import json

        name = response.get("name", "")
        args_raw = response.get("arguments", "{}")
        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        else:
            args = args_raw
        return registry.execute(name, **args)
