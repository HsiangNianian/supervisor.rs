"""SQLite database query tool."""

from __future__ import annotations

import sqlite3
from typing import Any

from supervisor.tools.base import BaseTool


class DatabaseTool(BaseTool):
    """Execute SQL queries against a SQLite database.

    Args:
        db_path: Path to the SQLite database file, or ':memory:'.
        read_only: If True, only SELECT statements are allowed.
    """

    name = "database"
    description = "Execute SQL queries against a SQLite database."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "SQL query to execute.",
            },
            "db_path": {
                "type": "string",
                "description": "Path to SQLite database (default: ':memory:').",
                "default": ":memory:",
            },
        },
        "required": ["query"],
    }

    def __init__(
        self, *, db_path: str = ":memory:", read_only: bool = False
    ) -> None:
        self.db_path = db_path
        self.read_only = read_only

    def execute(
        self,
        *,
        query: str,
        db_path: str | None = None,
        **_: Any,
    ) -> dict:
        """Execute a SQL query.

        Args:
            query: SQL query string.
            db_path: Optional override for database path.

        Returns:
            Dict with 'columns' and 'rows', or 'error'.
        """
        db = db_path or self.db_path
        stripped = query.strip().upper()

        if self.read_only and not stripped.startswith("SELECT"):
            return {"error": "Only SELECT queries are allowed in read-only mode."}

        try:
            conn = sqlite3.connect(db)
            cursor = conn.execute(query)
            if cursor.description:
                columns = [d[0] for d in cursor.description]
                rows = [list(row) for row in cursor.fetchall()]
                result = {"columns": columns, "rows": rows}
            else:
                conn.commit()
                result = {"affected_rows": cursor.rowcount}
            conn.close()
            return result
        except Exception as exc:
            return {"error": str(exc)}
