"""Typed runtime events emitted by the MiniHarness engine layer.

These events are frontend-neutral.  CLI, TUI, future web UI, logs, and tests
can subscribe to the same stream and render it however they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ReadyRuntimeEvent:
    type: Literal["ready"] = "ready"
    model: str = ""
    cwd: str = ""
    session_id: str = ""


@dataclass(frozen=True)
class AssistantDeltaEvent:
    type: Literal["assistant_delta"] = "assistant_delta"
    text: str = ""


@dataclass(frozen=True)
class AssistantCompleteEvent:
    type: Literal["assistant_complete"] = "assistant_complete"
    text: str = ""


@dataclass(frozen=True)
class ToolStartedEvent:
    type: Literal["tool_started"] = "tool_started"
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCompletedEvent:
    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str = ""
    output: str = ""
    is_error: bool = False
    artifact_path: str = ""
    original_output_chars: int = 0


@dataclass(frozen=True)
class PermissionRequestEvent:
    type: Literal["permission_request"] = "permission_request"
    request_id: str = ""
    tool_name: str = ""
    prompt: str = ""


@dataclass(frozen=True)
class ErrorRuntimeEvent:
    type: Literal["error"] = "error"
    message: str = ""
    recoverable: bool = True


@dataclass(frozen=True)
class StatusRuntimeEvent:
    type: Literal["status"] = "status"
    message: str = ""


@dataclass(frozen=True)
class SystemRuntimeEvent:
    type: Literal["system_message"] = "system_message"
    message: str = ""


@dataclass(frozen=True)
class TokenUsageRuntimeEvent:
    type: Literal["token_usage"] = "token_usage"
    token_count: int = 0
    context_window: int = 0
    soft_limit: int = 0
    usage_ratio: float = 0.0
    message_tokens: int = 0
    tool_tokens: int = 0
    response_reserve_tokens: int = 0
    available: int = 0
    tokenizer: str = ""
    model: str = ""


@dataclass(frozen=True)
class CompactProgressRuntimeEvent:
    type: Literal["compact_progress"] = "compact_progress"
    phase: str = ""
    tier: str = ""
    token_count: int = 0
    soft_limit: int = 0
    usage_ratio: float = 0.0
    compacted: bool = False
    tokens_after: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LineCompleteEvent:
    type: Literal["line_complete"] = "line_complete"


@dataclass(frozen=True)
class ShutdownRuntimeEvent:
    type: Literal["shutdown"] = "shutdown"


RuntimeEvent = (
    ReadyRuntimeEvent
    | AssistantDeltaEvent
    | AssistantCompleteEvent
    | ToolStartedEvent
    | ToolCompletedEvent
    | PermissionRequestEvent
    | ErrorRuntimeEvent
    | StatusRuntimeEvent
    | SystemRuntimeEvent
    | TokenUsageRuntimeEvent
    | CompactProgressRuntimeEvent
    | LineCompleteEvent
    | ShutdownRuntimeEvent
)
