"""Frontend-backend protocol — JSON-lines messages over stdin/stdout.

Every line on stdout is prefixed with ``MHJSON:`` followed by a JSON
object.  Lines without the prefix are treated as raw log output.

Mirrors OpenHarness's OHJSON protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Literal

# Protocol line prefix — used to distinguish structured messages from log noise.
PROTOCOL_PREFIX = "MHJSON:"


# ═══════════════════════════════════════════════════════════════════════════
# Backend → Frontend events
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AssistantDelta:
    """Streaming token from the LLM."""
    type: Literal["assistant_delta"] = "assistant_delta"
    text: str = ""


@dataclass
class AssistantComplete:
    """Final assistant message (after streaming)."""
    type: Literal["assistant_complete"] = "assistant_complete"
    text: str = ""


@dataclass
class ToolStarted:
    """Tool execution begins."""
    type: Literal["tool_started"] = "tool_started"
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCompleted:
    """Tool execution finished."""
    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str = ""
    output: str = ""
    is_error: bool = False


@dataclass
class PermissionRequest:
    """Backend asks frontend for permission to execute a tool."""
    type: Literal["permission_request"] = "permission_request"
    request_id: str = ""
    tool_name: str = ""
    prompt: str = ""  # human-readable question


@dataclass
class UserQuestionRequest:
    """Backend asks frontend to answer an agent follow-up question."""
    type: Literal["user_question_request"] = "user_question_request"
    request_id: str = ""
    question: str = ""


@dataclass
class ErrorEvent:
    """Error surfaced to the frontend."""
    type: Literal["error"] = "error"
    message: str = ""


@dataclass
class ReadyEvent:
    """Backend initialized and ready."""
    type: Literal["ready"] = "ready"
    model: str = ""
    cwd: str = ""
    session_id: str = ""


@dataclass
class StateSnapshot:
    """Observable runtime state snapshot."""
    type: Literal["state_snapshot"] = "state_snapshot"
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSnapshot:
    """UI-safe task representation."""
    id: str
    type: str
    status: str
    description: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class TasksSnapshot:
    """Observable task-list snapshot."""
    type: Literal["tasks_snapshot"] = "tasks_snapshot"
    tasks: list[TaskSnapshot] = field(default_factory=list)


@dataclass
class ShutdownEvent:
    """Backend is shutting down."""
    type: Literal["shutdown"] = "shutdown"


@dataclass
class StatusEvent:
    """Transient status message."""
    type: Literal["status"] = "status"
    message: str = ""


@dataclass
class TokenUsageEvent:
    """Current context/token budget snapshot."""
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


@dataclass
class CompactProgressEvent:
    """Conversation compaction progress."""
    type: Literal["compact_progress"] = "compact_progress"
    phase: str = ""
    tier: str = ""
    token_count: int = 0
    soft_limit: int = 0
    usage_ratio: float = 0.0
    compacted: bool = False
    tokens_after: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMessage:
    """Transcript-visible system message."""
    type: Literal["system_message"] = "system_message"
    message: str = ""


@dataclass
class LineComplete:
    """One submitted line finished processing."""
    type: Literal["line_complete"] = "line_complete"


BackendEvent = (
    AssistantDelta | AssistantComplete | ToolStarted | ToolCompleted
    | PermissionRequest | UserQuestionRequest | ErrorEvent | ReadyEvent
    | StateSnapshot | TasksSnapshot | ShutdownEvent | SystemMessage | LineComplete
    | StatusEvent | TokenUsageEvent | CompactProgressEvent
)


# ═══════════════════════════════════════════════════════════════════════════
# Frontend → Backend requests
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SubmitLine:
    """User submitted a prompt."""
    type: Literal["submit_line"] = "submit_line"
    line: str = ""


@dataclass
class PermissionResponse:
    """User answered a permission request."""
    type: Literal["permission_response"] = "permission_response"
    request_id: str = ""
    allowed: bool = False


@dataclass
class UserQuestionResponse:
    """User answered an agent follow-up question."""
    type: Literal["user_question_response"] = "user_question_response"
    request_id: str = ""
    answer: str = ""


@dataclass
class Interrupt:
    """User pressed Ctrl+C."""
    type: Literal["interrupt"] = "interrupt"


@dataclass
class FrontendShutdown:
    """User wants to exit."""
    type: Literal["shutdown"] = "shutdown"


FrontendRequest = SubmitLine | PermissionResponse | UserQuestionResponse | Interrupt | FrontendShutdown


# ═══════════════════════════════════════════════════════════════════════════
# Wire helpers
# ═══════════════════════════════════════════════════════════════════════════

import json


def encode_event(event: BackendEvent) -> str:
    """Serialize a backend event to a protocol line."""
    d = _asdict(event)
    return PROTOCOL_PREFIX + json.dumps(d, ensure_ascii=False, default=str)


def encode_request(req: FrontendRequest) -> str:
    """Serialize a frontend request to a protocol line."""
    d = _asdict(req)
    return json.dumps(d, ensure_ascii=False)


def decode_message(line: str) -> dict | None:
    """Parse a protocol line into a dict.

    Returns ``None`` if the line is not a valid protocol message.
    """
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def decode_event(line: str) -> dict | None:
    """Parse an event line (with MHJSON: prefix) into a dict."""
    line = line.strip()
    if not line.startswith(PROTOCOL_PREFIX):
        return None
    return decode_message(line[len(PROTOCOL_PREFIX):])


def _asdict(obj) -> dict:
    """Convert nested dataclasses to dicts, skipping None values."""
    return _to_jsonable(obj)


def _to_jsonable(value):
    if is_dataclass(value):
        result = {}
        for f in value.__dataclass_fields__:
            item = getattr(value, f)
            if item is not None:
                result[f] = _to_jsonable(item)
        return result
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _to_jsonable(item)
            for key, item in value.items()
            if item is not None
        }
    return value
