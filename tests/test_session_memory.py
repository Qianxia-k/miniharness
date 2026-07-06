from pathlib import Path

from miniharness.context.carryover import build_compact_attachments, init_tool_metadata
from miniharness.messages import Message
from miniharness.services.session_memory import (
    get_session_memory_content,
    get_session_memory_path,
    session_memory_to_compact_text,
    update_session_memory_file,
)
from miniharness.config.settings import Settings
from miniharness.ui.runtime import RuntimeController


def test_session_memory_file_round_trip(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIHARNESS_DATA_DIR", str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    metadata = init_tool_metadata()
    metadata["task_focus_state"]["goal"] = "finish session memory"
    messages = [
        Message(role="system", content="system"),
        Message(role="user", content="please finish memory runtime"),
        Message(role="assistant", content="done"),
    ]

    path = update_session_memory_file(
        project,
        messages,
        tool_metadata=metadata,
        session_id="abc",
    )

    assert path == get_session_memory_path(project, "abc")
    content = get_session_memory_content(path)
    assert "# Session Memory" in content
    assert "finish session memory" in content
    assert "please finish memory runtime" in content
    assert metadata["session_memory_path"] == str(path)
    assert metadata["session_id"] == "abc"


def test_session_memory_compact_attachment_uses_checkpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIHARNESS_DATA_DIR", str(tmp_path / "data"))
    project = tmp_path / "project"
    project.mkdir()
    metadata = init_tool_metadata()
    path = update_session_memory_file(
        project,
        [Message(role="user", content="remember this checkpoint")],
        tool_metadata=metadata,
        session_id="session-1",
    )

    attachments = build_compact_attachments(metadata)

    assert path.exists()
    assert any(
        attachment["content"].startswith("[Compact attachment: session_memory]")
        and "remember this checkpoint" in attachment["content"]
        for attachment in attachments
    )


def test_session_memory_to_compact_text_ignores_empty_content():
    assert session_memory_to_compact_text("") == ""
    assert "Session memory checkpoint" in session_memory_to_compact_text("# Session Memory\nok")


def test_runtime_checkpoints_session_memory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINIHARNESS_DATA_DIR", str(tmp_path / "data"))
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    runtime.loop.conversation.append(Message(role="user", content="build checkpoint"))

    try:
        runtime.loop._prepare_session_memory()
        runtime.loop._update_session_memory()
    finally:
        import asyncio

        asyncio.run(runtime.close())

    path = Path(runtime.loop.tool_metadata["session_memory_path"])
    assert path.exists()
    assert runtime.loop.session_id in path.name
    assert "build checkpoint" in path.read_text(encoding="utf-8")
