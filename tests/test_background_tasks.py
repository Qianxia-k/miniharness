import asyncio
from pathlib import Path

import pytest

from miniharness.permissions import PermissionChecker
from miniharness.tasks import BackgroundTaskManager, reset_background_task_manager_for_tests
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
