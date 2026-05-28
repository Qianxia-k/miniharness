from pathlib import Path

from miniharness.permissions import PermissionChecker
from miniharness.tool_registry import create_default_registry


def test_default_registry_has_all_tools(tmp_path: Path):
    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))

    assert registry.get("read_file") is not None
    assert registry.get("ls") is not None
    assert registry.get("grep") is not None
    assert registry.get("write_file") is not None
    assert registry.get("edit_file") is not None
    assert registry.get("bash") is not None
    assert registry.get("web_fetch") is not None
    assert registry.get("task") is not None


def test_unknown_tool(tmp_path: Path):
    """Executing an unknown tool returns an error."""
    import asyncio

    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))
    result = asyncio.run(registry.execute("nonexistent", {}))
    assert result.is_error is True
    assert "Unknown tool" in result.output
