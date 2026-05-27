"""OpenAI-compatible LLM client with async streaming and retry.

This module owns the provider-facing API call. MiniHarness uses async streaming
for all model calls, mirroring OpenHarness's SupportsStreamingMessages protocol.

Retry policy (mirrors OpenHarness):
    - Max 3 retries (4 attempts total).
    - Exponential backoff: 1s -> 2s -> 4s + random jitter, capped at 30s.
    - Retryable status codes: 429, 500, 502, 503, 529.
    - Connection errors and timeouts are also retried.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from typing import Any, AsyncIterator

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from rich.console import Console

from miniharness.messages import Message
from miniharness.providers import ProviderProfile


# ---------------------------------------------------------------------------
# Stream event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Base type for stream events."""


@dataclass
class TextDelta(StreamEvent):
    """A piece of text from the model, printed immediately for the typing effect."""

    text: str


@dataclass
class StreamComplete(StreamEvent):
    """Streaming finished. Carries the full accumulated message (text + tool calls)."""

    message: Message


# ---------------------------------------------------------------------------
# Non-streaming response (kept for backward compatibility).
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """Normalized model response used by the agent loop."""

    message: Message


# ---------------------------------------------------------------------------
# LLM client.
# ---------------------------------------------------------------------------

_stderr = Console(stderr=True)


class LLMClient:
    """Async wrapper around an OpenAI-compatible chat completion client."""

    _RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 529})
    _MAX_RETRIES: int = 3
    _BASE_DELAY: float = 1.0
    _MAX_DELAY: float = 30.0

    def __init__(
        self,
        *,
        profile: ProviderProfile,
        model: str,
        base_url: str | None = None,
    ) -> None:
        self.profile = profile
        self.model = model
        self.base_url = base_url or profile.base_url

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.profile.resolve_api_key(), base_url=self.base_url
        )

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if stream:
            params["stream"] = True
        if tools:
            params["tools"] = tools
        if self.profile.extra_body:
            params["extra_body"] = dict(self.profile.extra_body)
        return params

    def _is_retryable(self, error: Exception) -> bool:
        """Return True for transient errors worth retrying."""
        if isinstance(error, APIStatusError):
            return error.status_code in self._RETRYABLE_STATUSES
        return isinstance(error, (APIConnectionError, APITimeoutError))

    def _retry_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter.  Attempt is 0-indexed."""
        return min(
            self._BASE_DELAY * (2**attempt) + random.uniform(0, 1),
            self._MAX_DELAY,
        )

    async def _retry_sleep(self, attempt: int) -> None:
        """Async sleep for retry backoff."""
        await asyncio.sleep(self._retry_delay(attempt))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def stream(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion asynchronously, yielding text deltas then a final
        StreamComplete.

        Retries the initial connection with exponential backoff.
        """
        client = self._create_client()
        params = self._build_params(messages, tools, stream=True)

        # --- retry loop for the initial connection ---
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response_stream = await client.chat.completions.create(**params)
                break
            except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
                if attempt == self._MAX_RETRIES or not self._is_retryable(exc):
                    raise
                _stderr.print(
                    f"[yellow]API error (attempt {attempt + 1}/{self._MAX_RETRIES + 1}), "
                    f"retrying in {self._retry_delay(attempt):.1f}s: {exc}[/yellow]"
                )
                await self._retry_sleep(attempt)

        # --- consume the stream ---
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}

        async for chunk in response_stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta is None:
                continue

            if delta.content:
                content_parts.append(delta.content)
                yield TextDelta(text=delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = tool_calls_acc[idx]
                    if tc_delta.id:
                        acc["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            acc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            acc["function"]["arguments"] += tc_delta.function.arguments

        # --- assemble the final message ---
        tool_calls = list(tool_calls_acc.values()) if tool_calls_acc else None
        content = "".join(content_parts) if content_parts else ""

        yield StreamComplete(
            message=Message(
                role="assistant",
                content=content,
                tool_calls=tool_calls,
            )
        )
