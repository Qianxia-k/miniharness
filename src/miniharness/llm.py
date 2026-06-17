"""OpenAI-compatible LLM client with async streaming, retry, and PTL handling.

Production-grade features (Round 2 + 3):

- **PTL (Prompt Too Long) detection**: raises ``PromptTooLongError`` so the
  agent loop can trigger reactive compaction and retry.
- **Completion-token-limit renegotiation**: if the model rejects ``max_tokens``
  as too high, auto-reduces and retries.
- **Rich event hierarchy** (Round 3.1): ``TextDelta``, ``StreamComplete``,
  ``ToolExecutionStarted``, ``ToolExecutionCompleted``, ``ErrorEvent``,
  ``StatusEvent``, ``CompactProgressEvent`` — all flow through the same
  async iterator so the caller can observe every lifecycle moment.
"""

from __future__ import annotations

import asyncio
import random
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from rich.console import Console

from miniharness.config.settings import AgentSettings
from miniharness.messages import Message
from miniharness.providers import ProviderProfile


# ---------------------------------------------------------------------------
# Stream event types (Round 3.1 — rich hierarchy)
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Base type for all stream events."""


@dataclass
class TextDelta(StreamEvent):
    """A piece of text from the model, printed immediately for the typing effect."""

    text: str


@dataclass
class StreamComplete(StreamEvent):
    """Streaming finished. Carries the full accumulated message (text + tool calls)."""

    message: Message


@dataclass
class ToolExecutionStarted(StreamEvent):
    """The harness is about to execute a tool call."""

    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolExecutionCompleted(StreamEvent):
    """A tool has finished executing."""

    tool_name: str
    output: str
    is_error: bool = False


@dataclass
class ErrorEvent(StreamEvent):
    """An error surfaced to the user."""

    message: str
    recoverable: bool = True


@dataclass
class StatusEvent(StreamEvent):
    """A transient system status message."""

    message: str


class CompactPhase(str, Enum):
    """Phases of the compaction lifecycle (for CompactProgressEvent)."""

    HOOKS_START = "hooks_start"
    CONTEXT_COLLAPSE_START = "context_collapse_start"
    CONTEXT_COLLAPSE_END = "context_collapse_end"
    SESSION_MEMORY_START = "session_memory_start"
    SESSION_MEMORY_END = "session_memory_end"
    COMPACT_START = "compact_start"
    COMPACT_RETRY = "compact_retry"
    COMPACT_END = "compact_end"
    COMPACT_FAILED = "compact_failed"


@dataclass
class CompactProgressEvent(StreamEvent):
    """Structured progress event for conversation compaction."""

    phase: CompactPhase
    trigger: str  # "auto", "manual", "reactive"
    message: str | None = None
    attempt: int | None = None


# ---------------------------------------------------------------------------
# Non-streaming response (kept for backward compatibility).
# ---------------------------------------------------------------------------


@dataclass
class LLMResponse:
    """Normalized model response used by the agent loop."""

    message: Message


# ---------------------------------------------------------------------------
# PTL (Prompt Too Long) exception
# ---------------------------------------------------------------------------


class PromptTooLongError(Exception):
    """Raised when the API rejects a request because the prompt is too long.

    The agent loop catches this, triggers reactive compaction, and retries.
    """

    def __init__(self, message: str, original_error: Exception | None = None) -> None:
        super().__init__(message)
        self.original_error = original_error


class CompletionTokenLimitError(Exception):
    """Raised when the model rejects ``max_tokens`` as too high.

    The agent loop catches this, reduces max_tokens, and retries.
    """

    def __init__(self, message: str, supported_limit: int | None = None) -> None:
        super().__init__(message)
        self.supported_limit = supported_limit


# ---------------------------------------------------------------------------
# Error classification helpers
# ---------------------------------------------------------------------------

_PTL_NEEDLES: tuple[str, ...] = (
    "prompt too long",
    "context_length_exceeded",
    "context length",
    "maximum context",
    "context window",
    "input tokens exceed",
    "messages resulted in",
    "reduce the length of the messages",
    "configured limit",
    "too many tokens",
    "too large for the model",
    "maximum context length",
    "exceed_context",
    "exceeds the available context size",
    "available context size",
)


def is_prompt_too_long_error(exc: Exception) -> bool:
    """Return True if the exception indicates the prompt exceeded the context window."""
    text = str(exc).lower()
    return any(needle in text for needle in _PTL_NEEDLES)


def is_completion_token_limit_error(exc: Exception) -> bool:
    """Return True if the exception indicates max_tokens was rejected."""
    text = str(exc).lower()
    return ("max_tokens" in text or "max_completion_tokens" in text) and (
        "too large" in text or "at most" in text or "completion tokens" in text
    )


def extract_completion_token_limit(exc: Exception) -> int | None:
    """Try to extract the supported max_tokens limit from the error message."""
    text = str(exc).lower().replace(",", "")
    patterns = (
        r"supports at most\s+(\d+)\s+completion tokens",
        r"at most\s+(\d+)\s+completion tokens",
        r"max(?:imum)?(?:_completion)?[_\s-]tokens.*?(?:<=|less than or equal to|at most)\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return max(1, int(match.group(1)))
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------

_stderr = Console(stderr=True)


class LLMClient:
    """Async wrapper around an OpenAI-compatible chat completion client.

    Production-grade features:
    - Exponential backoff retry (429, 500, 502, 503, 529, connection errors)
    - PTL (Prompt Too Long) detection → raises ``PromptTooLongError``
    - Completion-token-limit renegotiation → raises ``CompletionTokenLimitError``
    - Rich event stream (TextDelta, ToolExecutionStarted, StreamComplete, etc.)
    """

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
        agent_settings: AgentSettings | None = None,
    ) -> None:
        self.profile = profile
        self.model = model
        self.base_url = base_url or profile.base_url
        # Shared mutable reference — runtime changes are visible immediately.
        self.agent_settings = agent_settings or AgentSettings()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.profile.resolve_api_key(),
            base_url=self.base_url,
            timeout=self.agent_settings.request_timeout,
        )

    def _build_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        stream: bool,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "messages": messages}
        if stream:
            params["stream"] = True
        if tools:
            params["tools"] = tools
        if self.profile.extra_body:
            params["extra_body"] = dict(self.profile.extra_body)
        # Forward LLM sampling params when explicitly set (None = use API default).
        if self.agent_settings.temperature is not None:
            params["temperature"] = self.agent_settings.temperature
        if self.agent_settings.top_p is not None:
            params["top_p"] = self.agent_settings.top_p
        # max_tokens_override takes precedence (used for completion-token renegotiation).
        if max_tokens_override is not None:
            params["max_tokens"] = max_tokens_override
        elif self.agent_settings.max_tokens is not None:
            params["max_tokens"] = self.agent_settings.max_tokens
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
        max_tokens_override: int | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a completion asynchronously.

        Yields :class:`TextDelta` events for each content chunk, then a final
        :class:`StreamComplete` with the assembled message.

        Raises :class:`PromptTooLongError` if the prompt exceeds the context
        window (caller should compact and retry).

        Raises :class:`CompletionTokenLimitError` if max_tokens is too high
        (caller should reduce and retry).
        """
        client = self._create_client()
        params = self._build_params(messages, tools, stream=True,
                                     max_tokens_override=max_tokens_override)

        # --- retry loop for the initial connection ---
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response_stream = await client.chat.completions.create(**params)
                break
            except (APIStatusError, APIConnectionError, APITimeoutError) as exc:
                # Check for PTL / completion-token-limit before retry decision.
                if is_prompt_too_long_error(exc):
                    raise PromptTooLongError(
                        f"Prompt exceeds context window: {exc}", original_error=exc
                    ) from exc
                if is_completion_token_limit_error(exc):
                    limit = extract_completion_token_limit(exc)
                    raise CompletionTokenLimitError(
                        f"max_tokens too high (supported: {limit}): {exc}",
                        supported_limit=limit,
                    ) from exc

                if attempt == self._MAX_RETRIES or not self._is_retryable(exc):
                    raise
                delay = self._retry_delay(attempt)
                _stderr.print(
                    f"[yellow]API error (attempt {attempt + 1}/{self._MAX_RETRIES + 1}), "
                    f"retrying in {delay:.1f}s: {exc}[/yellow]"
                )
                await asyncio.sleep(delay)

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
