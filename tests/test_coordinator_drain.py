import asyncio
from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.context.carryover import record_tool_carryover
from miniharness.messages import Message
from miniharness.sessions import load_latest_session
from miniharness.swarm.permission_sync import (
    SwarmPermissionRequest,
    read_pending_permissions,
    write_permission_request,
)
from miniharness.tasks import BackgroundTaskManager, reset_background_task_manager_for_tests
from miniharness.ui.coordinator_drain import (
    collect_completed_background_tasks,
    drain_completed_background_tasks,
    format_completed_background_task_notifications,
    pending_background_task_entries,
    wait_for_completed_async_agent_entries,
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
    assert any(
        message.role == "user" and "runtime-done" in (message.content or "")
        for message in runtime.loop.conversation.messages
    )

    latest = load_latest_session(str(tmp_path))
    assert latest is not None
    assert any(
        item["role"] == "user" and "runtime-done" in str(item.get("content") or "")
        for item in latest["messages"]
    )


@pytest.mark.asyncio
async def test_local_agent_completion_uses_task_notification_envelope(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    task = await manager.create_agent_task(
        command="cat >/dev/null; printf 'agent-result\\n'",
        prompt="review current diff",
        description="review file changes",
        cwd=tmp_path,
    )
    task.metadata["agent_id"] = "reviewer@default"
    metadata: dict = {}
    record_tool_carryover(
        metadata,
        tool_name="agent",
        arguments={
            "description": "review file changes",
            "prompt": "review current diff",
            "subagent_type": "reviewer",
        },
        result_output=f"Spawned agent reviewer@default (task_id={task.id}, backend=local_agent)",
        is_error=False,
    )
    await _wait_for_status(manager, task.id, "completed")

    completed = await wait_for_completed_async_agent_entries(metadata, manager=manager)
    message = format_completed_background_task_notifications(completed, manager=manager)

    assert "<task-notification>" in message
    assert "<task-id>reviewer@default</task-id>" in message
    assert "<status>completed</status>" in message
    assert "<result>agent-result</result>" in message
    assert "background-task-notification" not in message


@pytest.mark.asyncio
async def test_runtime_auto_submits_local_agent_notification_to_parent_loop(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    task = await manager.create_agent_task(
        command="cat >/dev/null; printf 'review-output\\n'",
        prompt="review current diff",
        description="review current diff",
        cwd=tmp_path,
    )
    metadata = {"agent_id": "reviewer@default"}
    manager.update_task_metadata(task.id, metadata)
    await _wait_for_status(manager, task.id, "completed")

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    record_tool_carryover(
        runtime.loop.tool_metadata,
        tool_name="agent",
        arguments={
            "description": "review current diff",
            "prompt": "review current diff",
            "subagent_type": "reviewer",
        },
        result_output=f"Spawned agent reviewer@default (task_id={task.id}, backend=local_agent)",
        is_error=False,
    )
    system_messages: list[str] = []
    prompts: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        prompts.append(prompt)
        loop.conversation.append(Message(role="user", content=prompt))
        loop.conversation.append(Message(role="assistant", content="coordinator handled result"))
        return "coordinator handled result"

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

    assert len(prompts) == 1
    assert "<task-notification>" in prompts[0]
    assert "<result>review-output</result>" in prompts[0]
    assert any("Submitting background agent result" in msg for msg in system_messages)
    assert runtime.loop.tool_metadata["async_agent_tasks"][0]["notification_sent"] is True


@pytest.mark.asyncio
async def test_runtime_drains_agent_notifications_until_no_pending_agents(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    first_task = await manager.create_agent_task(
        command="cat >/dev/null; printf 'first-review\\n'",
        prompt="first review",
        description="first review",
        cwd=tmp_path,
    )
    manager.update_task_metadata(first_task.id, {"agent_id": "reviewer-1@default"})
    await _wait_for_status(manager, first_task.id, "completed")

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    record_tool_carryover(
        runtime.loop.tool_metadata,
        tool_name="agent",
        arguments={
            "description": "first review",
            "prompt": "first review",
            "subagent_type": "reviewer",
        },
        result_output=f"Spawned agent reviewer-1@default (task_id={first_task.id}, backend=local_agent)",
        is_error=False,
    )
    prompts: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        prompts.append(prompt)
        loop.conversation.append(Message(role="user", content=prompt))
        loop.conversation.append(Message(role="assistant", content=f"handled {len(prompts)}"))
        if len(prompts) == 1:
            second_task = await manager.create_agent_task(
                command="cat >/dev/null; printf 'second-review\\n'",
                prompt="second review",
                description="second review",
                cwd=tmp_path,
            )
            manager.update_task_metadata(second_task.id, {"agent_id": "reviewer-2@default"})
            record_tool_carryover(
                loop.tool_metadata,
                tool_name="agent",
                arguments={
                    "description": "second review",
                    "prompt": "second review",
                    "subagent_type": "reviewer",
                },
                result_output=(
                    "Spawned agent reviewer-2@default "
                    f"(task_id={second_task.id}, backend=local_agent)"
                ),
                is_error=False,
            )
        return "handled"

    async def print_system(message: str) -> None:
        pass

    try:
        assert await runtime.handle_line(
            "/history",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert len(prompts) == 2
    assert "first-review" in prompts[0]
    assert "second-review" in prompts[1]
    assert all(
        entry.get("notification_sent") is True
        for entry in runtime.loop.tool_metadata["async_agent_tasks"]
    )


@pytest.mark.asyncio
async def test_runtime_drains_worker_permissions_while_waiting_for_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "tasks")
    )
    task = await manager.create_agent_task(
        command="sleep 0.6; printf 'agent-after-permission\\n'",
        prompt="needs approval",
        description="worker waits for approval",
        cwd=tmp_path,
    )
    manager.update_task_metadata(task.id, {"agent_id": "worker@default"})

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    record_tool_carryover(
        runtime.loop.tool_metadata,
        tool_name="agent",
        arguments={
            "description": "worker waits for approval",
            "prompt": "needs approval",
            "subagent_type": "worker",
        },
        result_output=f"Spawned agent worker@default (task_id={task.id}, backend=local_agent)",
        is_error=False,
    )
    await write_permission_request(
        SwarmPermissionRequest(
            id="perm-while-waiting",
            worker_id="worker@default",
            worker_name="worker",
            team_name="default",
            tool_name="bash",
            description="Allow worker bash?",
            input={"command": "printf ok", "is_read_only": False},
        )
    )

    approvals: list[str] = []

    async def permission_prompt(tool_name: str, prompt: str) -> bool:
        approvals.append(tool_name)
        return True

    runtime.permission_prompt = permission_prompt
    prompts: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        prompts.append(prompt)
        loop.conversation.append(Message(role="user", content=prompt))
        return "handled"

    async def print_system(message: str) -> None:
        pass

    try:
        await runtime._drain_coordinator_notifications(
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert approvals == ["bash"]
    assert await read_pending_permissions("default") == []
    assert len(prompts) == 1
    assert "agent-after-permission" in prompts[0]
    assert runtime.loop.tool_metadata["async_agent_tasks"][0]["notification_sent"] is True


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
