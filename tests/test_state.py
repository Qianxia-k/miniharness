from pathlib import Path

import pytest

from miniharness.config.settings import Settings
from miniharness.messages import Message
from miniharness.sessions import save_session_snapshot
from miniharness.state import AppState, AppStateStore
from miniharness.tasks import (
    BackgroundTaskManager,
    BackgroundTaskRecord,
    reset_background_task_manager_for_tests,
)
from miniharness.ui.runtime import RuntimeController


def test_app_state_store_updates_and_notifies_listeners():
    store = AppStateStore(AppState(model="m1", permission_mode="default", theme="default"))
    observed: list[AppState] = []

    unsubscribe = store.subscribe(observed.append)
    updated = store.set(model="m2", permission_mode="plan")

    assert updated.model == "m2"
    assert updated.permission_mode == "plan"
    assert store.get().model == "m2"
    assert observed == [updated]

    unsubscribe()
    store.set(model="m3")

    assert len(observed) == 1
    assert store.get().model == "m3"


def test_runtime_controller_initializes_observable_state(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())

    try:
        state = runtime.state_store.get()
    finally:
        import asyncio

        asyncio.run(runtime.close())

    assert state.model == runtime.loop.model
    assert state.permission_mode == "default"
    assert state.cwd == str(tmp_path.resolve())
    assert state.session_id == runtime.loop.session_id
    assert state.provider == runtime.settings.provider.name


def test_runtime_controller_exposes_task_snapshots(tmp_path: Path):
    manager = reset_background_task_manager_for_tests(
        BackgroundTaskManager(tasks_dir=tmp_path / "background")
    )
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    runtime.loop.task_manager.replace_all([
        {"content": "inspect repo", "status": "in_progress"},
    ])
    now = 1.0
    manager._tasks["bg-test"] = BackgroundTaskRecord(
        id="bg-test",
        type="local_bash",
        status="completed",
        description="run tests",
        cwd=str(tmp_path),
        output_file=tmp_path / "background" / "bg-test.log",
        command="pytest",
        created_at=now,
        started_at=now,
        ended_at=now,
        return_code=0,
    )

    try:
        snapshots = runtime.task_snapshots()
    finally:
        import asyncio

        asyncio.run(runtime.close())
        reset_background_task_manager_for_tests(
            BackgroundTaskManager(tasks_dir=tmp_path / "background-reset")
        )

    assert any(
        item.type == "session_task"
        and item.status == "in_progress"
        and item.description == "inspect repo"
        for item in snapshots
    )
    assert any(
        item.id == "bg-test"
        and item.type == "local_bash"
        and item.status == "completed"
        and item.metadata.get("return_code") == "0"
        for item in snapshots
    )


@pytest.mark.asyncio
async def test_runtime_state_updates_after_session_switch(tmp_path: Path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    target_id = "target-session"
    save_session_snapshot(
        cwd=str(tmp_path),
        model=runtime.loop.model,
        messages=[
            Message(role="system", content="system").to_openai(),
            Message(role="user", content="hello").to_openai(),
        ],
        session_id=target_id,
        session_state={},
    )

    try:
        result = runtime.commands.dispatch(f"/resume {target_id}", runtime._make_context())
        new_loop = getattr(runtime._last_context, "_new_loop", None)
        assert result.message is not None
        assert new_loop is not None
        await runtime._replace_loop(new_loop)
        state = runtime.state_store.get()
    finally:
        await runtime.close()

    assert runtime.loop.session_id == target_id
    assert state.session_id == target_id
    assert state.model == runtime.loop.model
