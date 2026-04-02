"""Web search tool using DuckDuckGo."""

from __future__ import annotations

from typing import Any

from supervisor.tools.base import BaseTool


class WebSearchTool(BaseTool):
    """Search the web using DuckDuckGo HTML.

    This is a lightweight implementation that does not require any
    external search SDK.
    """

    name = "web_search"
    description = "Search the web and return results with titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query.",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results to return.",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def execute(
        self, *, query: str, num_results: int = 5, **_: Any
    ) -> dict:
        """Search the web.

        Args:
            query: Search query string.
            num_results: Maximum number of results.

        Returns:
            Dict with 'results' list of {title, url, snippet}.
        """
        try:
            import httpx
        except ImportError:
            return {"error": "httpx is required for WebSearchTool"}

        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    "https://html.duckduckgo.com/html/",
                    params={"q": query},
                    headers={"User-Agent": "Mozilla/5.0"},
                )
                return self._parse_results(resp.text, num_results)
        except Exception as exc:
            return {"error": str(exc), "results": []}

    @staticmethod
    def _parse_results(html: str, limit: int) -> dict:
        """Parse DuckDuckGo HTML results."""
        import re

        results = []
        # Simple regex extraction of result links
        for match in re.finditer(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</span>',
            html,
            re.DOTALL,
        ):
            if len(results) >= limit:
                break
            url = match.group(1)
            title = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", match.group(3)).strip()
            results.append({"title": title, "url": url, "snippet": snippet})

        return {"results": results}
