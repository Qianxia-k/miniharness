"""File-based permission sync for delegated MiniHarness agents.

The protocol mirrors OpenHarness's swarm permission bridge:

1. A worker writes ``pending/{request_id}.json``.
2. The leader reads pending requests and asks the user/frontend.
3. The leader moves the request to ``resolved/{request_id}.json``.
4. The worker polls the resolved file and continues or denies the tool call.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import string
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from miniharness.permissions import PermissionDecision


AGENT_ID_ENV_VAR = "MINIHARNESS_AGENT_ID"
AGENT_NAME_ENV_VAR = "MINIHARNESS_AGENT_NAME"
AGENT_TEAM_ENV_VAR = "MINIHARNESS_AGENT_TEAM"

_READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {
        "read_file",
        "ls",
        "grep",
        "glob",
        "lsp",
        "sleep",
        "ask_user_question",
        "web_fetch",
        "task_list",
        "task_get",
        "task_output",
        "agent_list",
        "team_list",
        "memory_search",
        "memory_log",
        "mcp__list_resources",
        "mcp__read_resource",
    }
)


@dataclass(frozen=True)
class SwarmPermissionRequest:
    """Permission request forwarded from a worker to the leader."""

    id: str
    worker_id: str
    worker_name: str
    team_name: str
    tool_name: str
    description: str
    input: dict[str, Any]
    status: Literal["pending", "approved", "rejected"] = "pending"
    resolved_by: Literal["worker", "leader"] | None = None
    resolved_at: float | None = None
    feedback: str | None = None
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "worker_id": self.worker_id,
            "worker_name": self.worker_name,
            "team_name": self.team_name,
            "tool_name": self.tool_name,
            "description": self.description,
            "input": self.input,
            "status": self.status,
            "resolved_by": self.resolved_by,
            "resolved_at": self.resolved_at,
            "feedback": self.feedback,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SwarmPermissionRequest":
        return cls(
            id=str(data["id"]),
            worker_id=str(data.get("worker_id", data.get("workerId", ""))),
            worker_name=str(data.get("worker_name", data.get("workerName", ""))),
            team_name=str(data.get("team_name", data.get("teamName", ""))),
            tool_name=str(data.get("tool_name", data.get("toolName", ""))),
            description=str(data.get("description", "")),
            input=data.get("input") if isinstance(data.get("input"), dict) else {},
            status=data.get("status", "pending"),
            resolved_by=data.get("resolved_by", data.get("resolvedBy")),
            resolved_at=data.get("resolved_at", data.get("resolvedAt")),
            feedback=data.get("feedback"),
            created_at=float(data.get("created_at", data.get("createdAt", time.time()))),
        )


@dataclass(frozen=True)
class PermissionResolution:
    """Leader resolution for a pending permission request."""

    decision: Literal["approved", "rejected"]
    resolved_by: Literal["worker", "leader"] = "leader"
    feedback: str | None = None


def evaluate_permission_request(
    request: SwarmPermissionRequest,
    checker,
) -> PermissionDecision:
    """Evaluate a worker permission request with the leader's checker.

    This mirrors OpenHarness's swarm permission bridge: known read-only tools
    are approved without bothering the user, while mutating tools go through the
    same leader ``PermissionChecker`` that normal local tool calls use.
    """
    is_read_only = _request_is_read_only(request)
    file_path = (
        request.input.get("file_path")
        or request.input.get("path")
        or request.input.get("directory")
    )
    command = request.input.get("command")
    return checker.evaluate(
        tool_name=request.tool_name,
        file_path=str(file_path) if file_path else None,
        command=str(command) if command else None,
        is_read_only=is_read_only,
    )


def is_swarm_worker() -> bool:
    """Return whether the current process is a delegated worker."""
    return bool(os.environ.get(AGENT_ID_ENV_VAR) and os.environ.get(AGENT_TEAM_ENV_VAR))


def current_worker_identity() -> tuple[str, str, str]:
    """Return ``(agent_id, agent_name, team_name)`` from worker env vars."""
    agent_id = os.environ.get(AGENT_ID_ENV_VAR, "").strip()
    agent_name = os.environ.get(AGENT_NAME_ENV_VAR, "").strip() or agent_id
    team_name = os.environ.get(AGENT_TEAM_ENV_VAR, "").strip()
    if not agent_id or not team_name:
        raise ValueError("worker permission sync requires agent id and team env vars")
    return agent_id, agent_name, team_name


def generate_request_id() -> str:
    timestamp_ms = int(time.time() * 1000)
    rand = "".join(random.choices(string.ascii_lowercase + string.digits, k=7))
    return f"perm-{timestamp_ms}-{rand}"


def teams_root() -> Path:
    return Path.home() / ".miniharness" / "teams"


def get_permission_dir(team_name: str) -> Path:
    return teams_root() / _safe_team_name(team_name) / "permissions"


def _pending_dir(team_name: str) -> Path:
    return get_permission_dir(team_name) / "pending"


def _resolved_dir(team_name: str) -> Path:
    return get_permission_dir(team_name) / "resolved"


def _pending_path(team_name: str, request_id: str) -> Path:
    return _pending_dir(team_name) / f"{request_id}.json"


def _resolved_path(team_name: str, request_id: str) -> Path:
    return _resolved_dir(team_name) / f"{request_id}.json"


def _ensure_dirs(team_name: str) -> None:
    _pending_dir(team_name).mkdir(parents=True, exist_ok=True)
    _resolved_dir(team_name).mkdir(parents=True, exist_ok=True)


async def write_permission_request(
    request: SwarmPermissionRequest,
) -> SwarmPermissionRequest:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_write_permission_request, request)


def _sync_write_permission_request(
    request: SwarmPermissionRequest,
) -> SwarmPermissionRequest:
    _ensure_dirs(request.team_name)
    pending_path = _pending_path(request.team_name, request.id)
    tmp_path = pending_path.with_suffix(".json.tmp")
    with _exclusive_file_lock(_pending_dir(request.team_name) / ".lock"):
        tmp_path.write_text(
            json.dumps(request.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, pending_path)
    return request


async def read_pending_permissions(
    team_name: str | None = None,
) -> list[SwarmPermissionRequest]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_read_pending_permissions, team_name)


def _sync_read_pending_permissions(
    team_name: str | None = None,
) -> list[SwarmPermissionRequest]:
    team_names = [_safe_team_name(team_name)] if team_name else _discover_team_names()
    requests: list[SwarmPermissionRequest] = []
    for team in team_names:
        pending_dir = _pending_dir(team)
        if not pending_dir.exists():
            continue
        for path in sorted(pending_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                requests.append(SwarmPermissionRequest.from_dict(data))
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
    requests.sort(key=lambda item: item.created_at)
    return requests


async def resolve_permission(
    request_id: str,
    resolution: PermissionResolution,
    team_name: str,
) -> bool:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _sync_resolve_permission,
        request_id,
        resolution,
        _safe_team_name(team_name),
    )


def _sync_resolve_permission(
    request_id: str,
    resolution: PermissionResolution,
    team_name: str,
) -> bool:
    _ensure_dirs(team_name)
    pending_path = _pending_path(team_name, request_id)
    resolved_path = _resolved_path(team_name, request_id)
    tmp_path = resolved_path.with_suffix(".json.tmp")
    with _exclusive_file_lock(_pending_dir(team_name) / ".lock"):
        if not pending_path.exists():
            return False
        try:
            request = SwarmPermissionRequest.from_dict(
                json.loads(pending_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return False
        resolved = SwarmPermissionRequest(
            id=request.id,
            worker_id=request.worker_id,
            worker_name=request.worker_name,
            team_name=request.team_name,
            tool_name=request.tool_name,
            description=request.description,
            input=request.input,
            status="approved" if resolution.decision == "approved" else "rejected",
            resolved_by=resolution.resolved_by,
            resolved_at=time.time(),
            feedback=resolution.feedback,
            created_at=request.created_at,
        )
        tmp_path.write_text(
            json.dumps(resolved.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, resolved_path)
        try:
            pending_path.unlink()
        except OSError:
            pass
    return True


async def read_resolved_permission(
    request_id: str,
    team_name: str | None = None,
) -> SwarmPermissionRequest | None:
    team = _safe_team_name(team_name or os.environ.get(AGENT_TEAM_ENV_VAR, ""))
    if not team:
        return None
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_read_resolved_permission, request_id, team)


def _sync_read_resolved_permission(
    request_id: str,
    team_name: str,
) -> SwarmPermissionRequest | None:
    path = _resolved_path(team_name, request_id)
    if not path.exists():
        return None
    try:
        return SwarmPermissionRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return None


async def delete_resolved_permission(
    request_id: str,
    team_name: str | None = None,
) -> bool:
    team = _safe_team_name(team_name or os.environ.get(AGENT_TEAM_ENV_VAR, ""))
    if not team:
        return False
    path = _resolved_path(team, request_id)
    try:
        path.unlink()
        return True
    except (FileNotFoundError, OSError):
        return False


async def request_permission_from_leader(
    *,
    tool_name: str,
    tool_input: dict[str, Any],
    description: str,
    timeout_seconds: float = 300.0,
    poll_interval: float = 0.25,
) -> PermissionDecision:
    """Submit a worker permission request and wait for the leader response."""
    try:
        worker_id, worker_name, team_name = current_worker_identity()
    except ValueError as exc:
        return PermissionDecision(False, reason=str(exc))

    request = SwarmPermissionRequest(
        id=generate_request_id(),
        worker_id=worker_id,
        worker_name=worker_name,
        team_name=team_name,
        tool_name=tool_name,
        description=description,
        input=dict(tool_input),
    )
    await write_permission_request(request)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        resolved = await read_resolved_permission(request.id, team_name)
        if resolved is not None:
            await delete_resolved_permission(request.id, team_name)
            if resolved.status == "approved":
                return PermissionDecision(True)
            return PermissionDecision(False, reason=resolved.feedback or "User denied.")
        await asyncio.sleep(poll_interval)
    return PermissionDecision(False, reason="Timed out waiting for leader permission.")


def _discover_team_names() -> list[str]:
    root = teams_root()
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _safe_team_name(team_name: str | None) -> str:
    raw = (team_name or "").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in raw)


def _request_is_read_only(request: SwarmPermissionRequest) -> bool:
    if request.input.get("is_read_only") is True:
        return True
    return request.tool_name in _READ_ONLY_TOOLS


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        if os.name == "posix":
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        else:
            yield
    finally:
        handle.close()
