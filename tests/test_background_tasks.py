import asyncio
import json
import re
import shlex
import sys
from pathlib import Path

import pytest

from miniharness.permissions import PermissionChecker
from miniharness.context.carryover import build_compact_attachments, record_tool_carryover
from miniharness.tasks import (
    AgentRegistry,
    BackgroundTaskManager,
    reset_agent_registry_for_tests,
    reset_background_task_manager_for_tests,
    reset_team_registry_for_tests,
)
from miniharness.tasks.background import _default_agent_argv
from miniharness.tasks.worker_protocol import decode_worker_line, encode_worker_message
from miniharness.tool_registry import create_default_registry


async def _wait_for_status(manager: BackgroundTaskManager, task_id: str, *statuses: str):
    for _ in range(60):
        task = manager.get_task(task_id)
        if task is not None and task.status in statuses:
            return task
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} did not reach {statuses}")


@pytest.mark.asyncio
async def test_background_task_manager_runs_shell_and_captures_output(tmp_path: Path):
    manager = BackgroundTaskManager(tasks_dir=tmp_path / "tasks")

    task = await manager.create_shell_task(
        command="printf 'hello\\n'",
        description="print hello",
        cwd=tmp_path,
    )
    done = await _wait_for_status(manager, task.id, "completed")

    assert done.return_code == 0
    assert "hello" in manager.read_output(task.id)


@pytest.mark.asyncio
async def test_background_task_manager_persists_completed_records(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    manager = BackgroundTaskManager(tasks_dir=tasks_dir)

    task = await manager.create_shell_task(
        command="printf 'persisted\\n'",
        description="persist task",
        cwd=tmp_path,
    )
    await _wait_for_status(manager, task.id, "completed")

    restored = BackgroundTaskManager(tasks_dir=tasks_dir)
    restored_task = restored.get_task(task.id)

    assert restored_task is not None
    assert restored_task.status == "completed"
    assert restored_task.return_code == 0
    assert "persisted" in restored.read_output(task.id)


@pytest.mark.asyncio
async def test_background_task_manager_runs_local_agent_command_override(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    manager = BackgroundTaskManager(tasks_dir=tasks_dir)
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print('agent:' + sys.stdin.read().strip())")
    )

    task = await manager.create_agent_task(
        prompt="summarize this",
        description="agent smoke",
        cwd=tmp_path,
        command=command,
    )
    done = await _wait_for_status(manager, task.id, "completed")
    restored = BackgroundTaskManager(tasks_dir=tasks_dir).get_task(task.id)

    assert done.type == "local_agent"
    assert done.prompt == "summarize this"
    assert done.return_code == 0
    assert "agent:summarize this" in manager.read_output(task.id)
    assert restored is not None
    assert restored.type == "local_agent"
    assert restored.prompt == "summarize this"


def test_default_agent_argv_uses_task_worker_mode(tmp_path: Path):
    argv = _default_agent_argv(cwd=tmp_path, model="test-model")

    assert argv[:3] == [sys.executable, "-m", "miniharness"]
    assert "--cwd" in argv
    assert str(tmp_path.resolve()) in argv
    assert "--task-worker" in argv
    assert "--model" in argv
    assert "test-model" in argv


def test_worker_protocol_keeps_single_line_plain_and_frames_multiline():
    assert encode_worker_message("hello").decode("utf-8") == "hello\n"
    multiline = "line one\nline two"
    encoded = encode_worker_message(multiline).decode("utf-8")

    assert encoded.startswith('{"text":')
    assert encoded.endswith("\n")
    assert decode_worker_line(encoded) == multiline


def test_worker_protocol_preserves_structured_text_payload():
    payload = '{"text":"hello\\nworld","from":"coordinator"}'

    assert encode_worker_message(payload).decode("utf-8") == payload + "\n"
    assert decode_worker_line(payload + "\n") == "hello\nworld"


def test_background_task_manager_marks_orphaned_running_records_failed(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    output_file = tasks_dir / "bg-orphan.log"
    output_file.write_text("partial output\n", encoding="utf-8")
    (tasks_dir / "tasks.json").write_text(
        json.dumps({
            "version": 1,
            "tasks": [
                {
                    "id": "bg-orphan",
                    "type": "local_bash",
                    "status": "running",
                    "description": "orphaned",
                    "cwd": str(tmp_path),
                    "output_file": str(output_file),
                    "command": "sleep 30",
                    "created_at": 1.0,
                    "started_at": 1.0,
                    "metadata": {},
                }
            ],
        }),
        encoding="utf-8",
    )

    manager = BackgroundTaskManager(tasks_dir=tasks_dir)
    task = manager.get_task("bg-orphan")

    assert task is not None
    assert task.status == "failed"
    assert task.return_code == -1
    assert "not attached" in task.metadata["status_note"]
    assert "partial output" in manager.read_output("bg-orphan")


@pytest.mark.asyncio
async def test_background_task_tools_create_list_output_and_update(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    created = await registry.execute("task_create", {
        "type": "local_bash",
        "description": "print from background",
        "command": "printf 'background-ok\\n'",
    })
    assert created.is_error is False
    task_id = created.output.split()[3]

    await _wait_for_status(manager, task_id, "completed")

    listed = await registry.execute("task_list", {})
    assert task_id in listed.output
    assert "completed" in listed.output

    output = await registry.execute("task_output", {"task_id": task_id})
    assert output.is_error is False
    assert "background-ok" in output.output

    updated = await registry.execute("task_update", {
        "task_id": task_id,
        "progress": 100,
        "status_note": "verified",
    })
    assert updated.is_error is False
    assert "verified" in updated.output


@pytest.mark.asyncio
async def test_background_task_tool_creates_local_agent_with_command_override(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print('tool-agent:' + sys.stdin.read().strip())")
    )

    created = await registry.execute("task_create", {
        "type": "local_agent",
        "description": "agent from tool",
        "prompt": "inspect repo",
        "command": command,
    })
    assert created.is_error is False
    task_id = created.output.split()[3]

    await _wait_for_status(manager, task_id, "completed")
    output = await registry.execute("task_output", {"task_id": task_id})

    assert "local_agent" in created.output
    assert "tool-agent:inspect repo" in output.output


@pytest.mark.asyncio
async def test_agent_tool_spawns_pollable_local_agent_task(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print('agent-tool:' + sys.stdin.readline().strip())")
    )

    created = await registry.execute("agent", {
        "description": "agent from semantic tool",
        "prompt": "inspect repo",
        "subagent_type": "reviewer",
        "command": command,
    })

    assert created.is_error is False
    assert "Spawned agent reviewer@default" in created.output
    assert "backend=subprocess" in created.output
    match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert match, created.output
    task_id = match.group(1)

    done = await _wait_for_status(manager, task_id, "completed")
    output = await registry.execute("task_output", {"task_id": task_id})
    details = await registry.execute("task_get", {"task_id": task_id})

    assert done.type == "local_agent"
    assert "agent-tool:inspect repo" in output.output
    assert "reviewer: agent from semantic tool" in details.output
    assert done.metadata["agent_id"] == "reviewer@default"
    assert done.metadata["backend_type"] == "subprocess"


@pytest.mark.asyncio
async def test_agent_tool_allocates_unique_name_team_ids(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print(sys.stdin.readline().strip())")
    )

    first = await registry.execute("agent", {
        "description": "first reviewer",
        "prompt": "one",
        "subagent_type": "reviewer",
        "team": "qa",
        "command": command,
    })
    second = await registry.execute("agent", {
        "description": "second reviewer",
        "prompt": "two",
        "subagent_type": "reviewer",
        "team": "qa",
        "command": command,
    })

    assert first.is_error is False
    assert second.is_error is False
    assert "Spawned agent reviewer@qa" in first.output
    assert "Spawned agent reviewer-2@qa" in second.output
    for output in (first.output, second.output):
        match = re.search(r"task_id=(\S+?)[,)]", output)
        assert match, output
        await _wait_for_status(manager, match.group(1), "completed")


@pytest.mark.asyncio
async def test_agent_list_reports_agents_and_filters_team(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print(sys.stdin.readline().strip())")
    )

    first = await registry.execute("agent", {
        "description": "qa reviewer",
        "prompt": "one",
        "subagent_type": "reviewer",
        "team": "qa",
        "command": command,
    })
    second = await registry.execute("agent", {
        "description": "docs writer",
        "prompt": "two",
        "subagent_type": "writer",
        "team": "docs",
        "command": command,
    })
    for output in (first.output, second.output):
        match = re.search(r"task_id=(\S+?)[,)]", output)
        assert match, output
        await _wait_for_status(manager, match.group(1), "completed")

    listed = await registry.execute("agent_list", {})
    assert listed.is_error is False
    assert "reviewer@qa" in listed.output
    assert "writer@docs" in listed.output
    assert "status=completed" in listed.output

    qa_only = await registry.execute("agent_list", {"team": "qa"})
    assert "reviewer@qa" in qa_only.output
    assert "writer@docs" not in qa_only.output


@pytest.mark.asyncio
async def test_agent_list_restores_from_persisted_task_metadata(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tasks_dir)
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print(sys.stdin.readline().strip())")
    )

    created = await registry.execute("agent", {
        "description": "restored reviewer",
        "prompt": "inspect",
        "subagent_type": "reviewer",
        "team": "qa",
        "command": command,
    })
    match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert match, created.output
    await _wait_for_status(manager, match.group(1), "completed")

    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tasks_dir))
    reset_agent_registry_for_tests(AgentRegistry())
    restored_registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    listed = await restored_registry.execute("agent_list", {})
    assert listed.is_error is False
    assert "reviewer@qa" in listed.output
    assert "restored reviewer" in listed.output


@pytest.mark.asyncio
async def test_team_tools_create_list_and_delete_empty_team(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tasks_dir))
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    created = await registry.execute("team_create", {
        "name": "qa",
        "description": "Quality agents",
    })
    assert created.is_error is False
    assert "Created team qa" in created.output

    duplicate = await registry.execute("team_create", {"name": "qa"})
    assert duplicate.is_error is True
    assert "already exists" in duplicate.output

    listed = await registry.execute("team_list", {})
    assert listed.is_error is False
    assert "qa agents=0 description=Quality agents" in listed.output

    reset_team_registry_for_tests()
    persisted_list = await registry.execute("team_list", {})
    assert "qa agents=0 description=Quality agents" in persisted_list.output
    assert (tasks_dir / "teams.json").exists()

    deleted = await registry.execute("team_delete", {"name": "qa"})
    assert deleted.is_error is False
    assert "Deleted team qa" in deleted.output

    reset_team_registry_for_tests()
    listed_after = await registry.execute("team_list", {})
    assert listed_after.output == "(no teams)"


@pytest.mark.asyncio
async def test_team_list_restores_members_from_agent_tasks_and_delete_blocks_nonempty(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    command = (
        f"{shlex.quote(sys.executable)} -c "
        + shlex.quote("import sys; print(sys.stdin.readline().strip())")
    )

    created = await registry.execute("agent", {
        "description": "qa reviewer",
        "prompt": "inspect",
        "subagent_type": "reviewer",
        "team": "qa",
        "command": command,
    })
    match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert match, created.output
    task_id = match.group(1)
    await _wait_for_status(manager, task_id, "completed")

    reset_agent_registry_for_tests(AgentRegistry())
    reset_team_registry_for_tests()
    listed = await registry.execute("team_list", {})

    assert listed.is_error is False
    assert "qa agents=1" in listed.output
    assert f"reviewer@qa task_id={task_id} status=completed" in listed.output

    deleted = await registry.execute("team_delete", {"name": "qa"})
    assert deleted.is_error is True
    assert "not empty" in deleted.output


@pytest.mark.asyncio
async def test_task_stop_stops_agent_backing_task_and_preserves_agent_history(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    reset_team_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    script = "\n".join([
        "import sys",
        "print(sys.stdin.readline().strip(), flush=True)",
        "for line in sys.stdin:",
        "    pass",
    ])
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(script)

    created = await registry.execute("agent", {
        "description": "running worker",
        "prompt": "start",
        "subagent_type": "worker",
        "team": "qa",
        "command": command,
    })
    task_match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert task_match, created.output
    task_id = task_match.group(1)

    stopped = await registry.execute("task_stop", {"task_id": task_id})
    assert stopped.is_error is False
    assert f"Stopped background task {task_id} (killed)" in stopped.output

    listed = await registry.execute("agent_list", {})
    assert f"worker@qa task_id={task_id} status=killed" in listed.output
    assert manager.get_task(task_id).metadata["agent_id"] == "worker@qa"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_send_message_writes_to_running_agent_task(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    script = "\n".join([
        "import sys",
        "first = sys.stdin.readline().strip()",
        "print('INIT:' + first, flush=True)",
        "for line in sys.stdin:",
        "    text = line.strip()",
        "    print('MSG:' + text, flush=True)",
        "    if text == 'stop':",
        "        break",
    ])
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(script)

    created = await registry.execute("agent", {
        "description": "interactive worker",
        "prompt": "initial prompt",
        "subagent_type": "worker",
        "command": command,
    })
    assert created.is_error is False
    agent_match = re.search(r"Spawned agent (\S+)", created.output)
    task_match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert agent_match and task_match, created.output
    agent_id = agent_match.group(1)
    task_id = task_match.group(1)

    for _ in range(60):
        output = manager.read_output(task_id)
        if "INIT:initial prompt" in output:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("agent did not receive initial prompt")

    sent = await registry.execute("send_message", {
        "task_id": agent_id,
        "message": "follow up",
    })
    assert sent.is_error is False
    assert f"task_id={task_id}" in sent.output

    for _ in range(60):
        output = manager.read_output(task_id)
        if "MSG:follow up" in output:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("agent did not receive follow-up message")

    await registry.execute("send_message", {"task_id": task_id, "message": "stop"})
    await _wait_for_status(manager, task_id, "completed")


@pytest.mark.asyncio
async def test_send_message_frames_multiline_as_single_worker_line(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    script = "\n".join([
        "import sys",
        "print('INIT:' + sys.stdin.readline().strip(), flush=True)",
        "raw = sys.stdin.readline().strip()",
        "print('RAW:' + raw, flush=True)",
    ])
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(script)

    created = await registry.execute("agent", {
        "description": "multiline worker",
        "prompt": "initial prompt",
        "subagent_type": "worker",
        "command": command,
    })
    task_match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert task_match, created.output
    task_id = task_match.group(1)

    sent = await registry.execute("send_message", {
        "task_id": "worker@default",
        "message": "line one\nline two",
    })
    assert sent.is_error is False

    for _ in range(60):
        output = manager.read_output(task_id)
        if 'RAW:{"text": "line one\\nline two"}' in output:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("multiline message was not framed as one worker line")

    await _wait_for_status(manager, task_id, "completed")


@pytest.mark.asyncio
async def test_send_message_restores_agent_route_from_persisted_task_metadata(tmp_path: Path):
    tasks_dir = tmp_path / "tasks"
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tasks_dir)
    )
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )
    script = "\n".join([
        "import sys",
        "first = sys.stdin.readline().strip()",
        "print('INIT:' + first, flush=True)",
        "for line in sys.stdin:",
        "    text = line.strip()",
        "    print('MSG:' + text, flush=True)",
        "    if text == 'stop':",
        "        break",
    ])
    command = f"{shlex.quote(sys.executable)} -u -c " + shlex.quote(script)

    created = await registry.execute("agent", {
        "description": "restartable worker",
        "prompt": "initial prompt",
        "subagent_type": "worker",
        "team": "qa",
        "command": command,
    })
    task_match = re.search(r"task_id=(\S+?)[,)]", created.output)
    assert task_match, created.output
    task_id = task_match.group(1)

    await registry.execute("send_message", {"task_id": task_id, "message": "stop"})
    await _wait_for_status(manager, task_id, "completed")

    restored_manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tasks_dir)
    )
    reset_agent_registry_for_tests(AgentRegistry())
    restored_registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    sent = await restored_registry.execute("send_message", {
        "task_id": "worker@qa",
        "message": "after restore",
    })
    assert sent.is_error is False
    assert f"task_id={task_id}" in sent.output

    for _ in range(60):
        output = restored_manager.read_output(task_id)
        if "INIT:after restore" in output:
            break
        await asyncio.sleep(0.05)
    else:
        raise AssertionError("restored agent route did not receive message")

    await restored_registry.execute("send_message", {"task_id": "worker@qa", "message": "stop"})
    await _wait_for_status(restored_manager, task_id, "completed")


@pytest.mark.asyncio
async def test_agent_tool_respects_plan_mode(tmp_path: Path):
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tmp_path / "tasks"))
    reset_agent_registry_for_tests()
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="plan"),
    )

    result = await registry.execute("agent", {
        "description": "blocked semantic agent",
        "prompt": "do work",
        "command": "printf nope",
    })

    assert result.is_error is True
    assert "Read-only mode" in result.output


@pytest.mark.asyncio
async def test_send_message_respects_plan_mode(tmp_path: Path):
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tmp_path / "tasks"))
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="plan"),
    )

    result = await registry.execute("send_message", {
        "task_id": "bg-nope",
        "message": "do work",
    })

    assert result.is_error is True
    assert "Read-only mode" in result.output


@pytest.mark.asyncio
async def test_background_task_tool_local_agent_respects_plan_mode(tmp_path: Path):
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tmp_path / "tasks"))
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="plan"),
    )

    result = await registry.execute("task_create", {
        "type": "local_agent",
        "description": "blocked agent",
        "prompt": "do work",
        "command": "printf nope",
    })

    assert result.is_error is True
    assert "Read-only mode" in result.output


@pytest.mark.asyncio
async def test_background_task_create_respects_plan_mode(tmp_path: Path):
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tmp_path / "tasks"))
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="plan"),
    )

    result = await registry.execute("task_create", {
        "type": "local_bash",
        "description": "should not run",
        "command": "printf no",
    })

    assert result.is_error is True
    assert "Read-only mode" in result.output


@pytest.mark.asyncio
async def test_background_task_create_rejects_probable_mermaid(tmp_path: Path):
    reset_background_task_manager_for_tests(BackgroundTaskManager(tasks_dir=tmp_path / "tasks"))
    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = await registry.execute("task_create", {
        "type": "local_bash",
        "description": "bad markup",
        "command": "C[CLI] --> L[AgentLoop]",
    })

    assert result.is_error is True
    assert "probable Markdown/Mermaid" in result.output
    assert not (tmp_path / "L[AgentLoop]").exists()


def test_background_task_carryover_survives_compaction_attachment():
    metadata: dict = {}

    record_tool_carryover(
        metadata,
        tool_name="task_create",
        arguments={
            "type": "local_bash",
            "description": "run slow tests",
            "command": "uv run pytest",
        },
        result_output="Created background task bg-abc12345 (local_bash)",
        is_error=False,
    )
    record_tool_carryover(
        metadata,
        tool_name="task_output",
        arguments={"task_id": "bg-abc12345"},
        result_output="tests still running\ncollected 149 items",
        is_error=False,
    )

    state = metadata["background_task_state"]
    assert state[0]["id"] == "bg-abc12345"
    assert state[0]["status"] == "running"
    assert "tests still running" in state[0]["last_output_preview"]

    attachments = build_compact_attachments(metadata)
    background = [
        item for item in attachments
        if "[Compact attachment: background_tasks]" in item["content"]
    ]
    assert background
    assert "bg-abc12345" in background[0]["content"]
    assert "tests still running" in background[0]["content"]


def test_agent_carryover_survives_compaction_attachment():
    metadata: dict = {}

    record_tool_carryover(
        metadata,
        tool_name="agent",
        arguments={
            "description": "review file changes",
            "prompt": "inspect current diff",
            "subagent_type": "reviewer",
        },
        result_output="Spawned agent reviewer@default (task_id=bg-abc12345, backend=local_agent)",
        is_error=False,
    )

    state = metadata["background_task_state"]
    assert state[0]["id"] == "bg-abc12345"
    assert state[0]["agent_id"] == "reviewer@default"
    assert state[0]["type"] == "local_agent"
    assert state[0]["subagent_type"] == "reviewer"

    attachments = build_compact_attachments(metadata)
    background = [
        item for item in attachments
        if "[Compact attachment: background_tasks]" in item["content"]
    ]
    assert background
    assert "reviewer@default (bg-abc12345)" in background[0]["content"]
    assert "review file changes" in background[0]["content"]


def test_send_message_carryover_survives_compaction_attachment():
    metadata: dict = {}

    record_tool_carryover(
        metadata,
        tool_name="send_message",
        arguments={"task_id": "agent-abc12345", "message": "please inspect tests"},
        result_output="Sent message to agent-abc12345 (task_id=bg-abc12345)",
        is_error=False,
    )

    state = metadata["background_task_state"]
    assert state[0]["id"] == "bg-abc12345"
    assert state[0]["last_message_preview"] == "please inspect tests"

    attachments = build_compact_attachments(metadata)
    background = [
        item for item in attachments
        if "[Compact attachment: background_tasks]" in item["content"]
    ]
    assert background
    assert "last message: please inspect tests" in background[0]["content"]
