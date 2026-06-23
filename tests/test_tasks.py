from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.sessions.storage import load_session_by_id, save_session_snapshot
from miniharness.ui.runtime import RuntimeController


@pytest.mark.asyncio
async def test_task_tool_updates_session_task_manager(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        result = await runtime.loop.tools.execute("task", {
            "tasks": [
                {"content": "Inspect current implementation", "status": "completed"},
                {"content": "Add task manager", "status": "in_progress"},
                {"content": "Run tests", "status": "pending"},
            ]
        })
    finally:
        await runtime.close()

    assert result.is_error is False
    assert "task-001" in result.output
    assert "Add task manager" in result.output
    assert runtime.loop.task_manager.summary() == {
        "total": 3,
        "pending": 1,
        "in_progress": 1,
        "completed": 1,
    }


@pytest.mark.asyncio
async def test_task_tool_rejects_multiple_in_progress_items(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        result = await runtime.loop.tools.execute("task", {
            "tasks": [
                {"content": "One", "status": "in_progress"},
                {"content": "Two", "status": "in_progress"},
            ]
        })
    finally:
        await runtime.close()

    assert result.is_error is True
    assert "Only one task may be in_progress" in result.output


def test_task_tool_schema_inlines_nested_items(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        task_tool_schema = runtime.loop.tools.get("task").to_openai_tool()
    finally:
        # No async resources have been started, so this is enough.
        pass

    tasks_schema = task_tool_schema["function"]["parameters"]["properties"]["tasks"]
    assert tasks_schema["type"] == "array"
    assert "$ref" not in str(tasks_schema)
    assert "content" in tasks_schema["items"]["properties"]


@pytest.mark.asyncio
async def test_session_switch_restores_task_state_without_cross_contamination(tmp_path: Path):
    save_session_snapshot(
        cwd=str(tmp_path),
        model="gpt",
        messages=[{"role": "user", "content": "session a"}],
        session_id="a",
        session_state={
            "tool_metadata": {
                "task_list_state": {
                    "tasks": [
                        {"id": "task-001", "content": "A only task", "status": "in_progress"}
                    ],
                    "revision": 1,
                    "updated_at": 1.0,
                }
            }
        },
    )
    save_session_snapshot(
        cwd=str(tmp_path),
        model="gpt",
        messages=[{"role": "user", "content": "session b"}],
        session_id="b",
        session_state={
            "tool_metadata": {
                "task_list_state": {
                    "tasks": [
                        {"id": "task-001", "content": "B only task", "status": "pending"}
                    ],
                    "revision": 1,
                    "updated_at": 1.0,
                }
            }
        },
    )

    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    runtime.loop.session_id = "a"
    runtime.loop.restore_messages([{"role": "user", "content": "session a"}])
    runtime.loop.restore_session_state(load_session_by_id(str(tmp_path), "a")["session_state"])
    system_messages: list[str] = []

    async def run_agent(loop, prompt: str) -> str:
        raise AssertionError("agent should not run for /resume")

    async def print_system(message: str) -> None:
        system_messages.append(message)

    try:
        assert await runtime.handle_line(
            "/resume b",
            run_agent=run_agent,
            print_system=print_system,
        )
    finally:
        await runtime.close()

    assert [task.content for task in runtime.loop.task_manager.list_tasks()] == ["B only task"]
    saved_a = load_session_by_id(str(tmp_path), "a")
    assert saved_a is not None
    saved_tasks = saved_a["session_state"]["tool_metadata"]["task_list_state"]["tasks"]
    assert saved_tasks[0]["content"] == "A only task"
