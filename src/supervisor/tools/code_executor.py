"""Sandboxed code execution tool."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any

from supervisor.tools.base import BaseTool


class CodeExecutorTool(BaseTool):
    """Execute code in a sandboxed subprocess with timeout.

    Currently supports Python. The code is written to a temporary file
    and executed in a subprocess with a configurable timeout.

    Args:
        default_timeout: Default execution timeout in seconds.
    """

    name = "code_executor"
    description = "Execute Python code in a sandboxed subprocess."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Source code to execute.",
            },
            "language": {
                "type": "string",
                "description": "Programming language (currently only 'python').",
                "default": "python",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds.",
                "default": 10,
            },
        },
        "required": ["code"],
    }

    def __init__(self, *, default_timeout: int = 10) -> None:
        self.default_timeout = default_timeout

    def execute(
        self,
        *,
        code: str,
        language: str = "python",
        timeout: int | None = None,
        **_: Any,
    ) -> dict:
        """Execute code and return stdout/stderr/exit_code.

        Args:
            code: Source code string.
            language: Programming language.
            timeout: Execution timeout in seconds.

        Returns:
            Dict with 'stdout', 'stderr', and 'exit_code'.
        """
        if language != "python":
            return {"stdout": "", "stderr": f"Unsupported language: {language}", "exit_code": -1}

        timeout = timeout or self.default_timeout
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                ["python3", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "stderr": f"Execution timed out after {timeout}s",
                "exit_code": -1,
            }
        finally:
            Path(tmp_path).unlink(missing_ok=True)
