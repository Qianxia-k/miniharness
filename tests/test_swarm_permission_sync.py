import asyncio
from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.permissions import PermissionChecker
from miniharness.swarm.permission_sync import (
    AGENT_ID_ENV_VAR,
    AGENT_NAME_ENV_VAR,
    AGENT_TEAM_ENV_VAR,
    PermissionResolution,
    SwarmPermissionRequest,
    evaluate_permission_request,
    read_resolved_permission,
    read_pending_permissions,
    resolve_permission,
    write_permission_request,
)
from miniharness.tool_registry import create_default_registry
from miniharness.ui.runtime import RuntimeController


@pytest.mark.asyncio
async def test_permission_sync_file_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    request = SwarmPermissionRequest(
        id="perm-test",
        worker_id="worker@default",
        worker_name="worker",
        team_name="default",
        tool_name="write_file",
        description="Allow write",
        input={"path": "a.txt", "is_read_only": False},
    )

    await write_permission_request(request)
    pending = await read_pending_permissions("default")

    assert [item.id for item in pending] == ["perm-test"]

    resolved = await resolve_permission(
        "perm-test",
        PermissionResolution(decision="approved"),
        "default",
    )
    pending_after = await read_pending_permissions("default")

    assert resolved is True
    assert pending_after == []


@pytest.mark.asyncio
async def test_worker_tool_permission_waits_for_leader_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv(AGENT_ID_ENV_VAR, "worker@default")
    monkeypatch.setenv(AGENT_NAME_ENV_VAR, "worker")
    monkeypatch.setenv(AGENT_TEAM_ENV_VAR, "default")
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="default"),
    )

    async def leader_resolver():
        for _ in range(80):
            pending = await read_pending_permissions("default")
            if pending:
                await resolve_permission(
                    pending[0].id,
                    PermissionResolution(decision="approved"),
                    "default",
                )
                return
            await asyncio.sleep(0.05)
        raise AssertionError("worker did not write a permission request")

    resolver_task = asyncio.create_task(leader_resolver())
    result = await registry.execute("write_file", {
        "path": "approved.txt",
        "content": "ok",
    })
    await resolver_task

    assert result.is_error is False
    assert (tmp_path / "approved.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_runtime_drains_pending_swarm_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    approvals: list[tuple[str, str]] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        approvals.append((tool_name, prompt))
        return True

    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        permission_prompt=permission_prompt,
    )
    request = SwarmPermissionRequest(
        id="perm-runtime",
        worker_id="worker@default",
        worker_name="worker",
        team_name="default",
        tool_name="bash",
        description="Allow bash to run command: printf ok?",
        input={"command": "printf ok", "is_read_only": False},
    )
    await write_permission_request(request)

    resolved = await runtime._drain_swarm_permission_requests(lambda message: asyncio.sleep(0))
    pending = await read_pending_permissions("default")

    assert resolved == 1
    assert pending == []
    assert approvals
    assert approvals[0][0] == "bash"


def test_permission_request_evaluator_auto_allows_read_only_tool(tmp_path: Path):
    request = SwarmPermissionRequest(
        id="perm-read",
        worker_id="worker@default",
        worker_name="worker",
        team_name="default",
        tool_name="read_file",
        description="Read README",
        input={"path": str(tmp_path / "README.md")},
    )

    decision = evaluate_permission_request(
        request,
        PermissionChecker(cwd=tmp_path, mode="default"),
    )

    assert decision.allowed is True
    assert decision.requires_confirmation is False


@pytest.mark.asyncio
async def test_runtime_auto_resolves_read_only_worker_permission_without_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    approvals: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        approvals.append(tool_name)
        return False

    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        permission_prompt=permission_prompt,
    )
    request = SwarmPermissionRequest(
        id="perm-read-runtime",
        worker_id="worker@default",
        worker_name="worker",
        team_name="default",
        tool_name="read_file",
        description="Read project file",
        input={"path": str(tmp_path / "README.md")},
    )
    await write_permission_request(request)

    resolved_count = await runtime._drain_swarm_permission_requests(
        lambda message: asyncio.sleep(0)
    )
    resolved = await read_resolved_permission("perm-read-runtime", "default")

    assert resolved_count == 1
    assert approvals == []
    assert resolved is not None
    assert resolved.status == "approved"


@pytest.mark.asyncio
async def test_runtime_denies_plan_mode_worker_mutation_without_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    approvals: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        approvals.append(tool_name)
        return True

    runtime = RuntimeController(
        cwd=tmp_path,
        settings=Settings(),
        permission_prompt=permission_prompt,
        permission_mode="plan",
    )
    request = SwarmPermissionRequest(
        id="perm-plan-write",
        worker_id="worker@default",
        worker_name="worker",
        team_name="default",
        tool_name="write_file",
        description="Write project file",
        input={"path": str(tmp_path / "out.txt"), "is_read_only": False},
    )
    await write_permission_request(request)

    resolved_count = await runtime._drain_swarm_permission_requests(
        lambda message: asyncio.sleep(0)
    )
    resolved = await read_resolved_permission("perm-plan-write", "default")

    assert resolved_count == 1
    assert approvals == []
    assert resolved is not None
    assert resolved.status == "rejected"
    assert resolved.feedback == "Read-only mode (plan)"
