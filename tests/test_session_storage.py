"""Tests for session persistence."""

from pathlib import Path

import pytest

from miniharness.sessions.storage import (
    _project_slug,
    load_latest_session,
    load_session_by_id,
    load_session_by_tag,
    mark_session_latest,
    rename_session,
    save_session_snapshot,
)


@pytest.fixture(autouse=True)
def isolate_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


class TestProjectSlug:
    def test_slug_is_deterministic(self):
        a = _project_slug("/home/user/my-project")
        b = _project_slug("/home/user/my-project")
        assert a == b

    def test_slug_differs_by_path(self):
        a = _project_slug("/home/user/project-a")
        b = _project_slug("/home/user/project-b")
        assert a != b


class TestSaveAndLoad:
    def test_save_and_load_latest(self, tmp_path: Path):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]
        save_session_snapshot(
            cwd=str(tmp_path),
            model="test-model",
            messages=messages,
            session_id="abc123",
        )

        data = load_latest_session(str(tmp_path))
        assert data is not None
        assert data["session_id"] == "abc123"
        assert data["model"] == "test-model"
        assert len(data["messages"]) == 2
        assert data["summary"] == "Hello"

    def test_load_by_id(self, tmp_path: Path):
        messages = [{"role": "user", "content": "Hi"}]
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=messages,
            session_id="xyz789",
        )

        data = load_session_by_id(str(tmp_path), "xyz789")
        assert data is not None
        assert data["session_id"] == "xyz789"

    def test_update_preserves_created_at(self, tmp_path: Path):
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=[{"role": "user", "content": "first"}],
            session_id="same-id",
        )
        first = load_session_by_id(str(tmp_path), "same-id")

        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=[{"role": "user", "content": "first"}, {"role": "assistant", "content": "ok"}],
            session_id="same-id",
        )
        second = load_session_by_id(str(tmp_path), "same-id")

        assert first is not None
        assert second is not None
        assert second["created_at"] == first["created_at"]
        assert second["updated_at"] >= first["updated_at"]

    def test_load_nonexistent(self, tmp_path: Path):
        assert load_latest_session(str(tmp_path)) is None
        assert load_session_by_id(str(tmp_path), "nope") is None
        assert load_session_by_id(str(tmp_path), "../nope") is None

    def test_rejects_unsafe_session_id_on_save(self, tmp_path: Path):
        with pytest.raises(ValueError):
            save_session_snapshot(
                cwd=str(tmp_path),
                model="gpt",
                messages=[{"role": "user", "content": "Hi"}],
                session_id="../escape",
            )

    def test_switching_sessions_does_not_cross_write_messages(self, tmp_path: Path):
        a_messages = [{"role": "user", "content": "A only"}]
        b_messages = [{"role": "user", "content": "B only"}]

        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=a_messages,
            session_id="session-a",
        )
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=b_messages,
            session_id="session-b",
        )

        b_after_chat = b_messages + [{"role": "assistant", "content": "B reply"}]
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=b_after_chat,
            session_id="session-b",
        )

        data_a = load_session_by_id(str(tmp_path), "session-a")
        data_b = load_session_by_id(str(tmp_path), "session-b")

        assert data_a is not None
        assert data_b is not None
        assert data_a["messages"] == a_messages
        assert data_b["messages"] == b_after_chat
        assert load_latest_session(str(tmp_path))["session_id"] == "session-b"

    def test_save_without_make_latest_does_not_move_latest_pointer(self, tmp_path: Path):
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=[{"role": "user", "content": "latest"}],
            session_id="latest-session",
        )
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=[{"role": "user", "content": "background save"}],
            session_id="background-session",
            make_latest=False,
        )

        assert load_latest_session(str(tmp_path))["session_id"] == "latest-session"

    def test_mark_session_latest_does_not_change_messages(self, tmp_path: Path):
        messages = [{"role": "user", "content": "target"}]
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=[{"role": "user", "content": "other"}],
            session_id="other",
        )
        save_session_snapshot(
            cwd=str(tmp_path),
            model="gpt",
            messages=messages,
            session_id="target",
            make_latest=False,
        )

        assert mark_session_latest(str(tmp_path), "target")
        assert load_latest_session(str(tmp_path))["session_id"] == "target"
        assert load_session_by_id(str(tmp_path), "target")["messages"] == messages


class TestListSessions:
    def test_list_sessions(self, tmp_path: Path):
        from miniharness.sessions.storage import list_sessions

        save_session_snapshot(
            cwd=str(tmp_path),
            model="a",
            messages=[{"role": "user", "content": "first"}],
            session_id="001",
        )
        save_session_snapshot(
            cwd=str(tmp_path),
            model="b",
            messages=[{"role": "user", "content": "second"}],
            session_id="002",
        )

        sessions = list_sessions(str(tmp_path))
        assert len(sessions) >= 2
        ids = {s["session_id"] for s in sessions}
        assert "001" in ids
        assert "002" in ids


class TestRenameAndTag:
    def test_rename_and_load_by_tag(self, tmp_path: Path):
        messages = [{"role": "user", "content": "debug session"}]
        save_session_snapshot(
            cwd=str(tmp_path),
            model="test",
            messages=messages,
            session_id="sid001",
        )

        assert rename_session(str(tmp_path), "sid001", "my-debug")
        data = load_session_by_tag(str(tmp_path), "my-debug")
        assert data is not None
        assert data["session_id"] == "sid001"
        assert data["tag"] == "my-debug"

    def test_rename_nonexistent(self, tmp_path: Path):
        assert not rename_session(str(tmp_path), "no-such-id", "name")

    def test_load_by_tag_nonexistent(self, tmp_path: Path):
        assert load_session_by_tag(str(tmp_path), "no-such-tag") is None


class TestReplSessionSwitch:
    def test_resume_saves_current_before_switching_without_overwriting_target(self, tmp_path: Path):
        from miniharness.cli import _repl_resume
        from miniharness.config.settings import Settings
        from miniharness.loop import AgentLoop

        a_messages = [{"role": "user", "content": "A before switch"}]
        b_messages = [{"role": "user", "content": "B before switch"}]
        save_session_snapshot(cwd=str(tmp_path), model="gpt", messages=a_messages, session_id="a")
        save_session_snapshot(cwd=str(tmp_path), model="gpt", messages=b_messages, session_id="b")

        current_a = a_messages + [{"role": "assistant", "content": "A unsaved"}]

        loop = AgentLoop(cwd=tmp_path, settings=Settings())
        loop.session_id = "a"
        loop.restore_messages(current_a)

        next_loop = _repl_resume("b", loop)

        assert next_loop is not loop
        assert next_loop.session_id == "b"
        assert [
            {"role": msg["role"], "content": msg["content"]}
            for msg in next_loop.export_messages()
        ] == b_messages
        assert [
            {"role": msg["role"], "content": msg["content"]}
            for msg in load_session_by_id(str(tmp_path), "a")["messages"]
        ] == current_a
        assert load_session_by_id(str(tmp_path), "b")["messages"] == b_messages
        assert load_latest_session(str(tmp_path))["session_id"] == "b"
