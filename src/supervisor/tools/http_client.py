"""HTTP client tool."""

from __future__ import annotations

from typing import Any

from supervisor.tools.base import BaseTool


class HTTPClientTool(BaseTool):
    """Make HTTP requests.

    Args:
        default_timeout: Default request timeout in seconds.
    """

    name = "http_client"
    description = "Make HTTP requests to external APIs."
    parameters = {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                "description": "HTTP method.",
            },
            "url": {
                "type": "string",
                "description": "Request URL.",
            },
            "headers": {
                "type": "object",
                "description": "Request headers.",
            },
            "body": {
                "type": "string",
                "description": "Request body.",
            },
        },
        "required": ["method", "url"],
    }

    def __init__(self, *, default_timeout: int = 30) -> None:
        self.default_timeout = default_timeout

    def execute(
        self,
        *,
        method: str,
        url: str,
        headers: dict | None = None,
        body: str | None = None,
        **_: Any,
    ) -> dict:
        """Make an HTTP request.

        Args:
            method: HTTP method.
            url: Target URL.
            headers: Optional request headers.
            body: Optional request body.

        Returns:
            Dict with 'status_code', 'headers', 'body'.
        """
        try:
            import httpx
        except ImportError:
            return {"error": "httpx is required for HTTPClientTool"}

        try:
            with httpx.Client(timeout=self.default_timeout) as client:
                response = client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    content=body,
                )
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.text,
                }
        except Exception as exc:
            return {"error": str(exc)}
