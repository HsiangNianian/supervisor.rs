"""Tests for prompt template engine."""

from __future__ import annotations

import os

import pytest

from supervisor.prompt import PromptTemplate


class TestPromptTemplate:
    def test_simple_render(self):
        tpl = PromptTemplate("Hello, {{ name }}!")
        assert tpl.render(name="Alice") == "Hello, Alice!"

    def test_multiple_variables(self):
        tpl = PromptTemplate("{{ greeting }}, {{ name }}! You are a {{ role }}.")
        result = tpl.render(greeting="Hi", name="Bob", role="developer")
        assert result == "Hi, Bob! You are a developer."

    def test_variables_property(self):
        tpl = PromptTemplate("{{ a }} and {{ b }} and {{ c }}")
        assert tpl.variables == ["a", "b", "c"]

    def test_variables_sorted(self):
        tpl = PromptTemplate("{{ zebra }} {{ alpha }}")
        assert tpl.variables == ["alpha", "zebra"]

    def test_no_variables(self):
        tpl = PromptTemplate("No variables here.")
        assert tpl.variables == []
        assert tpl.render() == "No variables here."

    def test_missing_variable_raises(self):
        tpl = PromptTemplate("{{ name }} {{ age }}")
        with pytest.raises(ValueError, match="Missing required template variables"):
            tpl.render(name="Alice")

    def test_missing_multiple_variables_message(self):
        tpl = PromptTemplate("{{ a }} {{ b }} {{ c }}")
        with pytest.raises(ValueError, match="a") as exc_info:
            tpl.render(c="only_c")
        assert "b" in str(exc_info.value)

    def test_extra_variables_ignored(self):
        tpl = PromptTemplate("Hello, {{ name }}!")
        result = tpl.render(name="Alice", extra="ignored")
        assert result == "Hello, Alice!"

    def test_jinja2_features(self):
        tpl = PromptTemplate("{% for item in items %}{{ item }} {% endfor %}")
        result = tpl.render(items=["a", "b", "c"])
        assert result == "a b c "

    def test_conditional(self):
        tpl = PromptTemplate(
            "{% if formal %}Dear {{ name }}{% else %}Hey {{ name }}{% endif %}"
        )
        assert tpl.render(formal=True, name="Sir") == "Dear Sir"
        assert tpl.render(formal=False, name="Bob") == "Hey Bob"

    def test_invalid_syntax(self):
        with pytest.raises(ValueError, match="Invalid template syntax"):
            PromptTemplate("{% invalid %}")

    def test_repr(self):
        tpl = PromptTemplate("Hello, {{ name }}!")
        r = repr(tpl)
        assert "PromptTemplate" in r

    def test_repr_long_template(self):
        long = "x" * 100
        tpl = PromptTemplate(long)
        r = repr(tpl)
        assert "..." in r

    def test_multiline_template(self):
        tpl = PromptTemplate(
            "Line 1: {{ a }}\nLine 2: {{ b }}\nLine 3: {{ c }}"
        )
        result = tpl.render(a="1", b="2", c="3")
        assert "Line 1: 1" in result
        assert "Line 3: 3" in result


class TestPromptTemplateFromFile:
    def test_from_file(self, request):
        """Test loading template from a file."""
        # Create a test template file in the project directory.
        test_dir = os.path.dirname(os.path.abspath(__file__))
        tpl_path = os.path.join(test_dir, "_test_template.txt")
        try:
            with open(tpl_path, "w") as f:
                f.write("Hello, {{ name }}! Welcome to {{ place }}.")
            tpl = PromptTemplate.from_file(tpl_path)
            assert tpl.variables == ["name", "place"]
            assert tpl.render(name="Alice", place="Wonderland") == (
                "Hello, Alice! Welcome to Wonderland."
            )
        finally:
            if os.path.exists(tpl_path):
                os.remove(tpl_path)

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            PromptTemplate.from_file("/nonexistent/path/template.txt")
