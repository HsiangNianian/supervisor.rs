"""Tests for the built-in tool ecosystem."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from supervisor.tools.base import BaseTool
from supervisor.tools.calculator import CalculatorTool
from supervisor.tools.code_executor import CodeExecutorTool
from supervisor.tools.database import DatabaseTool
from supervisor.tools.file_io import FileIOTool
from supervisor.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# BaseTool
# ---------------------------------------------------------------------------

class TestBaseTool:
    def test_abstract(self):
        with pytest.raises(TypeError):
            BaseTool()

    def test_openai_spec(self):
        class DummyTool(BaseTool):
            name = "dummy"
            description = "A dummy tool"
            parameters = {"type": "object", "properties": {}}
            def execute(self, **kwargs):
                return "ok"

        spec = DummyTool().to_openai_spec()
        assert spec["type"] == "function"
        assert spec["function"]["name"] == "dummy"

    def test_to_schema(self):
        class DummyTool(BaseTool):
            name = "dummy"
            description = "desc"
            parameters = {}
            def execute(self, **kwargs):
                return None

        schema = DummyTool().to_schema()
        assert schema["name"] == "dummy"


# ---------------------------------------------------------------------------
# CalculatorTool
# ---------------------------------------------------------------------------

class TestCalculatorTool:
    def setup_method(self):
        self.calc = CalculatorTool()

    def test_basic_arithmetic(self):
        assert self.calc.execute(expression="2 + 3")["result"] == 5

    def test_multiplication(self):
        assert self.calc.execute(expression="4 * 5")["result"] == 20

    def test_division(self):
        assert self.calc.execute(expression="10 / 4")["result"] == 2.5

    def test_power(self):
        assert self.calc.execute(expression="2 ** 8")["result"] == 256

    def test_nested(self):
        assert self.calc.execute(expression="(2 + 3) * 4")["result"] == 20

    def test_unary_negative(self):
        assert self.calc.execute(expression="-5 + 3")["result"] == -2

    def test_safe_function_sqrt(self):
        result = self.calc.execute(expression="sqrt(16)")
        assert result["result"] == 4.0

    def test_invalid_expression(self):
        result = self.calc.execute(expression="import os")
        assert "error" in result

    def test_name(self):
        assert self.calc.name == "calculator"


# ---------------------------------------------------------------------------
# CodeExecutorTool
# ---------------------------------------------------------------------------

class TestCodeExecutorTool:
    def setup_method(self):
        self.executor = CodeExecutorTool(default_timeout=5)

    def test_simple_code(self):
        result = self.executor.execute(code="print('hello')")
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_syntax_error(self):
        result = self.executor.execute(code="def")
        assert result["exit_code"] != 0

    def test_unsupported_language(self):
        result = self.executor.execute(code="x", language="java")
        assert "Unsupported" in result["stderr"]

    def test_name(self):
        assert self.executor.name == "code_executor"


# ---------------------------------------------------------------------------
# FileIOTool
# ---------------------------------------------------------------------------

class TestFileIOTool:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fio = FileIOTool(base_dir=self.tmpdir)

    def test_write_and_read(self):
        self.fio.execute(action="write", path="test.txt", content="hello")
        result = self.fio.execute(action="read", path="test.txt")
        assert result["content"] == "hello"

    def test_exists(self):
        assert self.fio.execute(action="exists", path="nope.txt")["exists"] is False
        self.fio.execute(action="write", path="f.txt", content="x")
        assert self.fio.execute(action="exists", path="f.txt")["exists"] is True

    def test_list(self):
        self.fio.execute(action="write", path="a.txt", content="a")
        self.fio.execute(action="write", path="b.txt", content="b")
        result = self.fio.execute(action="list", path=".")
        assert "a.txt" in result["files"]

    def test_path_traversal_blocked(self):
        result = self.fio.execute(action="read", path="../../etc/passwd")
        assert "error" in result

    def test_name(self):
        assert self.fio.name == "file_io"


# ---------------------------------------------------------------------------
# DatabaseTool
# ---------------------------------------------------------------------------

class TestDatabaseTool:
    def test_create_and_query(self):
        db = DatabaseTool(db_path=":memory:")
        db.execute(query="CREATE TABLE t (id INTEGER, name TEXT)")
        db.execute(query="INSERT INTO t VALUES (1, 'alice')")
        result = db.execute(query="SELECT * FROM t")
        assert result["columns"] == ["id", "name"]
        assert result["rows"] == [[1, "alice"]]

    def test_read_only_blocks_insert(self):
        db = DatabaseTool(db_path=":memory:", read_only=True)
        result = db.execute(query="INSERT INTO t VALUES (1, 'x')")
        assert "error" in result

    def test_invalid_sql(self):
        db = DatabaseTool(db_path=":memory:")
        result = db.execute(query="INVALID SQL HERE")
        assert "error" in result

    def test_name(self):
        assert DatabaseTool().name == "database"


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------

class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        calc = CalculatorTool()
        reg.register(calc)
        assert reg.get("calculator") is calc

    def test_get_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("nope")

    def test_list_tools(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(DatabaseTool())
        assert len(reg.list_tools()) == 2

    def test_search(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        reg.register(DatabaseTool())
        results = reg.search("math")
        assert len(results) == 1
        assert results[0].name == "calculator"

    def test_get_openai_tools(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        specs = reg.get_openai_tools()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"

    def test_execute(self):
        reg = ToolRegistry()
        reg.register(CalculatorTool())
        result = reg.execute("calculator", expression="1+1")
        assert result["result"] == 2
