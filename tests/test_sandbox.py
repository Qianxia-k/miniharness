from pathlib import Path

from miniharness.sandbox.path_validator import validate_sandbox_path
from miniharness.sandbox.session import is_sandbox_active


class TestPathValidator:
    def test_path_inside_workspace(self, tmp_path: Path):
        allowed, reason = validate_sandbox_path(tmp_path / "foo.txt", tmp_path)
        assert allowed is True
        assert reason == ""

    def test_path_outside_workspace(self, tmp_path: Path):
        outside = Path("/tmp")
        allowed, reason = validate_sandbox_path(outside, tmp_path)
        assert allowed is False
        assert "outside sandbox boundary" in reason

    def test_path_equals_workspace(self, tmp_path: Path):
        allowed, reason = validate_sandbox_path(tmp_path, tmp_path)
        assert allowed is True

    def test_symlink_inside_workspace(self, tmp_path: Path):
        """Ensure resolved symlinks are properly validated."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        real_file = subdir / "real.txt"
        real_file.write_text("hello")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(real_file)

        allowed, reason = validate_sandbox_path(symlink, tmp_path)
        assert allowed is True


class TestSessionDefaults:
    def test_no_sandbox_by_default(self):
        """Without starting a session, sandbox is not active."""
        assert is_sandbox_active() is False
