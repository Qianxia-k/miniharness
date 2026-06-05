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
    assert registry.get("memory_search") is not None
    assert registry.get("memory_add") is not None
    assert registry.get("memory_log") is not None


def test_unknown_tool(tmp_path: Path):
    """Executing an unknown tool returns an error."""
    import asyncio

    registry = create_default_registry(cwd=tmp_path, permissions=PermissionChecker(cwd=tmp_path))
    result = asyncio.run(registry.execute("nonexistent", {}))
    assert result.is_error is True
    assert "Unknown tool" in result.output


def test_bash_rejects_mermaid_edge_before_shell_creates_redirect_file(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {
        "command": "C[CLI] --> L[AgentLoop]",
    }))

    assert result.is_error is True
    assert "probable Markdown/Mermaid" in result.output
    assert not (tmp_path / "L[AgentLoop]").exists()


def test_bash_allows_quoted_arrow_text(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {
        "command": "printf '%s\\n' 'C[CLI] --> L[AgentLoop]'",
    }))

    assert result.is_error is False
    assert "C[CLI] --> L[AgentLoop]" in result.output


def test_bash_allows_normal_command(tmp_path: Path):
    import asyncio

    registry = create_default_registry(
        cwd=tmp_path,
        permissions=PermissionChecker(cwd=tmp_path, mode="bypass"),
    )

    result = asyncio.run(registry.execute("bash", {"command": "printf ok"}))

    assert result.is_error is False
    assert result.output == "ok"
