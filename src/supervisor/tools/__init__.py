"""Built-in tool ecosystem for the supervisor agent framework."""

from __future__ import annotations

from supervisor.tools.base import BaseTool
from supervisor.tools.calculator import CalculatorTool
from supervisor.tools.code_executor import CodeExecutorTool
from supervisor.tools.database import DatabaseTool
from supervisor.tools.file_io import FileIOTool
from supervisor.tools.http_client import HTTPClientTool
from supervisor.tools.registry import ToolRegistry
from supervisor.tools.web_search import WebSearchTool

__all__ = [
    "BaseTool",
    "CalculatorTool",
    "CodeExecutorTool",
    "DatabaseTool",
    "FileIOTool",
    "HTTPClientTool",
    "ToolRegistry",
    "WebSearchTool",
]
