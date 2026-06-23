import asyncio
from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.context.carryover import record_tool_carryover
from miniharness.tasks import BackgroundTaskManager, reset_background_task_manager_for_tests
from miniharness.ui.coordinator_drain import (
    collect_completed_background_tasks,
    drain_completed_background_tasks,
    pending_background_task_entries,
)
from miniharness.ui.runtime import RuntimeController


async def _wait_for_status(manager: BackgroundTaskManager, task_id: str, *statuses: str):
    for _ in range(60):
        task = manager.get_task(task_id)
        if task is not None and task.status in statuses:
            return task
        await asyncio.sleep(0.05)
    raise AssertionError(f"task {task_id} did not reach {statuses}")


@pytest.mark.asyncio
async def test_drain_completed_background_task_notifies_once(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    task = await manager.create_shell_task(
        command="printf 'done\\n'",
        description="finish async work",
        cwd=tmp_path,
    )
    metadata: dict = {}
    record_tool_carryover(
        metadata,
        tool_name="task_create",
        arguments={
            "type": "local_bash",
            "description": task.description,
            "command": task.command,
        },
        result_output=f"Created background task {task.id} (local_bash)",
        is_error=False,
    )
    await _wait_for_status(manager, task.id, "completed")
    messages: list[str] = []

    async def print_system(message: str) -> None:
        messages.append(message)

    first = await drain_completed_background_tasks(
        metadata,
        print_system=print_system,
        manager=manager,
    )
    second = await drain_completed_background_tasks(
        metadata,
        print_system=print_system,
        manager=manager,
    )

    assert "background-task-notification" in first
    assert "done" in first
    assert second == ""
    assert len(messages) == 1
    assert metadata["background_task_state"][0]["notification_sent"] is True
    assert pending_background_task_entries(metadata) == []


@pytest.mark.asyncio
async def test_runtime_surfaces_completed_background_task_after_line(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    task = await manager.create_shell_task(
        command="printf 'runtime-done\\n'",
        description="runtime background",
        cwd=tmp_path,
    )
    await _wait_for_status(manager, task.id, "completed")

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    record_tool_carryover(
        runtime.loop.tool_metadata,
        tool_name="task_create",
        arguments={
            "type": "local_bash",
            "description": task.description,
            "command": task.command,
        },
        result_output=f"Created background task {task.id} (local_bash)",
        is_error=False,
    )
    system_messages: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        raise AssertionError("slash command should not call agent")

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        assert await runtime.handle_line(
            "/history",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert any("Conversation has" in message for message in system_messages)
    assert any("runtime-done" in message for message in system_messages)


def test_collect_completed_background_tasks_ignores_missing_and_marks_sent():
    metadata = {
        "background_task_state": [
            {"id": "bg-missing", "status": "running"},
            {"id": "", "status": "running"},
        ]
    }
    manager = BackgroundTaskManager(tasks_dir=Path("/tmp/miniharness-test-tasks"))

    completed = collect_completed_background_tasks(metadata, manager=manager)

    assert completed == []
    assert metadata["background_task_state"][0]["status"] == "missing"
    assert metadata["background_task_state"][0]["notification_sent"] is True
