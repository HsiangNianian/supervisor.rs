"""Tests for third-party tool integration adapters."""

import json

import pytest

from supervisor.tools.calculator import CalculatorTool
from supervisor.tools.integrations import LangChainToolAdapter, OpenAIFunctionAdapter
from supervisor.tools.registry import ToolRegistry


class FakeLangChainTool:
    """Minimal mock of a LangChain tool."""

    name = "fake_search"
    description = "Fake search tool"

    def _run(self, query: str = "") -> str:
        return f"results for: {query}"


class TestLangChainToolAdapter:
    def test_wrap_lc_tool(self):
        lc = FakeLangChainTool()
        adapter = LangChainToolAdapter(lc)
        assert adapter.name == "fake_search"
        result = adapter.execute(query="python")
        assert "python" in result

    def test_from_langchain(self):
        lc = FakeLangChainTool()
        adapter = LangChainToolAdapter.from_langchain(lc)
        assert adapter.name == "fake_search"

    def test_openai_spec(self):
        lc = FakeLangChainTool()
        adapter = LangChainToolAdapter(lc)
        spec = adapter.to_openai_spec()
        assert spec["function"]["name"] == "fake_search"


class TestOpenAIFunctionAdapter:
    def test_to_openai_functions(self):
        tools = [CalculatorTool()]
        specs = OpenAIFunctionAdapter.to_openai_functions(tools)
        assert len(specs) == 1
        assert specs[0]["function"]["name"] == "calculator"

    def test_call_from_openai_response(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())

        response = {
            "name": "calculator",
            "arguments": json.dumps({"expression": "3 + 4"}),
        }
        result = OpenAIFunctionAdapter.call_from_openai_response(response, reg)
        assert result["result"] == 7

    def test_call_with_dict_arguments(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())

        response = {
            "name": "calculator",
            "arguments": {"expression": "10 - 2"},
        }
        result = OpenAIFunctionAdapter.call_from_openai_response(response, reg)
        assert result["result"] == 8
