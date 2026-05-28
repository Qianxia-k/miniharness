"""Fetch content from a URL."""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class WebFetchInput(BaseModel):
    """Arguments for web_fetch."""

    url: str = Field(description="URL to fetch (must start with http:// or https://)")
    max_chars: int = Field(default=8000, description="Maximum characters to return")


class WebFetchTool(BaseTool):
    name = "web_fetch"
    description = (
        "Fetch the content of a URL and return it as plain text. "
        "Useful for reading documentation, API responses, or any web-accessible content. "
        "Returns a truncated view if the content exceeds max_chars."
    )
    input_model = WebFetchInput

    _TIMEOUT = 15.0
    _MAX_REDIRECTS = 3

    async def execute(self, arguments: WebFetchInput) -> ToolResult:
        url = arguments.url.strip()
        if not url:
            return ToolResult("url is required", is_error=True)
        if not re.match(r"^https?://", url):
            return ToolResult(f"Invalid URL: {url} (must start with http:// or https://)", is_error=True)

        try:
            async with httpx.AsyncClient(
                timeout=self._TIMEOUT,
                max_redirects=self._MAX_REDIRECTS,
                follow_redirects=True,
                headers={"User-Agent": "MiniHarness/1.0"},
            ) as client:
                response = await client.get(url)
                response.raise_for_status()
        except httpx.TimeoutException:
            return ToolResult(f"Timeout fetching {url} after {self._TIMEOUT}s", is_error=True)
        except httpx.TooManyRedirects:
            return ToolResult(f"Too many redirects for {url}", is_error=True)
        except httpx.HTTPStatusError as exc:
            return ToolResult(
                f"HTTP {exc.response.status_code} fetching {url}", is_error=True
            )
        except httpx.RequestError as exc:
            return ToolResult(f"Request failed: {exc}", is_error=True)

        text = _html_to_text(response.text)
        max_chars = max(1, min(arguments.max_chars, 50000))
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text) - max_chars} more chars)"

        return ToolResult(text)


# ---------------------------------------------------------------------------
# Minimal HTML → text converter (stdlib only, no BeautifulSoup dependency)
# ---------------------------------------------------------------------------

_STRIP_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def _html_to_text(html: str) -> str:
    """Strip HTML tags and return readable plain text."""
    text = _STRIP_RE.sub("", html)
    text = _STYLE_RE.sub("", text)
    text = _TAG_RE.sub("\n", text)
    text = _WS_RE.sub("\n\n", text)
    # Collapse leading / trailing whitespace on each line.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)
