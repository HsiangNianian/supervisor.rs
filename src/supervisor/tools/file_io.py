"""Sandboxed file I/O tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from supervisor.tools.base import BaseTool


class FileIOTool(BaseTool):
    """Read and write files within an allowed directory.

    All paths are resolved relative to ``base_dir`` and validated to
    prevent directory traversal attacks.

    Args:
        base_dir: Root directory for all file operations.
    """

    name = "file_io"
    description = "Read, write, and list files within a sandboxed directory."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "write", "list", "exists"],
                "description": "File operation to perform.",
            },
            "path": {
                "type": "string",
                "description": "Relative file path.",
            },
            "content": {
                "type": "string",
                "description": "Content to write (for 'write' action).",
            },
        },
        "required": ["action", "path"],
    }

    def __init__(self, *, base_dir: str = ".") -> None:
        self.base_dir = Path(base_dir).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve and validate a path against the base directory."""
        resolved = (self.base_dir / path).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def execute(
        self,
        *,
        action: str,
        path: str = ".",
        content: str = "",
        **_: Any,
    ) -> dict:
        """Perform a file operation.

        Args:
            action: One of 'read', 'write', 'list', 'exists'.
            path: Relative file path.
            content: Content for write operations.

        Returns:
            Dict with operation result.
        """
        try:
            resolved = self._resolve(path)

            if action == "read":
                return {"content": resolved.read_text()}
            elif action == "write":
                resolved.parent.mkdir(parents=True, exist_ok=True)
                resolved.write_text(content)
                return {"written": str(resolved.relative_to(self.base_dir))}
            elif action == "list":
                target = resolved if resolved.is_dir() else resolved.parent
                items = [str(p.relative_to(self.base_dir)) for p in target.iterdir()]
                return {"files": sorted(items)}
            elif action == "exists":
                return {"exists": resolved.exists()}
            else:
                return {"error": f"Unknown action: {action}"}
        except Exception as exc:
            return {"error": str(exc)}
