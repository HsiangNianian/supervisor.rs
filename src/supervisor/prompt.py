"""Jinja2-based prompt template engine.

Provides :class:`PromptTemplate` for rendering LLM prompts from
parameterised templates with variable validation.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import BaseLoader, Environment, TemplateSyntaxError, meta


class PromptTemplate:
    """Render prompts from Jinja2 templates with variable validation.

    Parameters:
        template: A Jinja2 template string.

    Example::

        tpl = PromptTemplate("Hello, {{ name }}! You are a {{ role }}.")
        print(tpl.render(name="Alice", role="developer"))
    """

    def __init__(self, template: str) -> None:
        self._raw = template
        self._env = Environment(
            loader=BaseLoader(),
            autoescape=False,
            keep_trailing_newline=True,
        )
        try:
            self._template = self._env.from_string(template)
        except TemplateSyntaxError as exc:
            raise ValueError(f"Invalid template syntax: {exc}") from exc

    @classmethod
    def from_file(cls, path: str) -> PromptTemplate:
        """Load a template from a file.

        Args:
            path: Filesystem path to the template file.

        Returns:
            A new :class:`PromptTemplate` instance.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        text = Path(path).read_text(encoding="utf-8")
        return cls(text)

    @property
    def variables(self) -> list[str]:
        """Return sorted list of undeclared variable names in the template."""
        ast = self._env.parse(self._raw)
        return sorted(meta.find_undeclared_variables(ast))

    def render(self, **kwargs: object) -> str:
        """Render the template with the given keyword arguments.

        Raises:
            ValueError: If required template variables are missing.
        """
        required = set(self.variables)
        provided = set(kwargs.keys())
        missing = required - provided
        if missing:
            raise ValueError(
                f"Missing required template variables: {', '.join(sorted(missing))}"
            )
        return self._template.render(**kwargs)

    def __repr__(self) -> str:
        preview = self._raw[:50] + ("..." if len(self._raw) > 50 else "")
        # Collapse whitespace in preview for readability.
        preview = re.sub(r"\s+", " ", preview)
        return f"PromptTemplate({preview!r})"


__all__ = ["PromptTemplate"]
