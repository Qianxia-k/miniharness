"""Filesystem globbing tool."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field

from miniharness.sandbox.session import get_sandbox
from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult


class GlobInput(BaseModel):
    """Arguments for glob."""

    pattern: str = Field(
        description="Glob pattern relative to the working directory",
        validation_alias=AliasChoices("pattern", "path"),
    )
    root: str | None = Field(default=None, description="Optional search root")
    limit: int = Field(default=200, ge=1, le=5000)


class GlobTool(BaseTool):
    """List files matching a glob pattern."""

    name = "glob"
    description = "List files matching a glob pattern."
    input_model = GlobInput

    def is_read_only(self, arguments: GlobInput) -> bool:
        del arguments
        return True

    def permission_requests(self, arguments: GlobInput) -> list[ToolPermissionRequest]:
        root, _ = _resolve_glob_request(self.cwd, arguments.root, arguments.pattern)
        return [ToolPermissionRequest(is_read_only=True, file_path=str(root))]

    async def execute(self, arguments: GlobInput) -> ToolResult:
        root, pattern = _resolve_glob_request(self.cwd, arguments.root, arguments.pattern)
        if not pattern.strip():
            return ToolResult("pattern is required", is_error=True)
        if not root.exists() or not root.is_dir():
            return ToolResult("(no matches)")

        matches = await _glob(root, pattern, limit=arguments.limit)
        if not matches:
            return ToolResult("(no matches)")
        return ToolResult("\n".join(matches))


def _resolve_path(base: Path, candidate: str | None) -> Path:
    path = Path(candidate or ".").expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _resolve_glob_request(base: Path, root_arg: str | None, pattern: str) -> tuple[Path, str]:
    """Return a concrete search root plus a root-relative glob pattern."""
    if not pattern.strip():
        return (_resolve_path(base, root_arg) if root_arg else base, pattern)

    candidate = Path(pattern).expanduser()
    if not candidate.is_absolute():
        return (_resolve_path(base, root_arg) if root_arg else base, pattern)

    parts = candidate.parts
    first_glob_index = next(
        (index for index, part in enumerate(parts) if _has_glob_magic(part)),
        None,
    )
    if first_glob_index is None:
        return candidate.parent.resolve(), candidate.name

    root_parts = parts[:first_glob_index]
    root = Path(*root_parts).resolve() if root_parts else Path(candidate.anchor or "/").resolve()
    relative_pattern = str(Path(*parts[first_glob_index:]))
    return root, relative_pattern


def _has_glob_magic(value: str) -> bool:
    return any(char in value for char in "*?[")


def _looks_like_git_repo(path: Path) -> bool:
    """Return whether hidden project dirs such as .github are probably relevant."""
    current = path
    for _ in range(6):
        if (current / ".git").exists():
            return True
        if current.parent == current:
            break
        current = current.parent
    return False


_GLOB_RG_TIMEOUT_SECONDS = 30.0


async def _glob(root: Path, pattern: str, *, limit: int) -> list[str]:
    """Fast glob implementation using ripgrep when available."""
    rg = shutil.which("rg")
    if rg and ("**" in pattern or "/" in pattern):
        include_hidden = _looks_like_git_repo(root)
        cmd = [rg, "--files"]
        if include_hidden:
            cmd.append("--hidden")
        cmd.extend(["--glob", pattern, "."])
        lines = await _run_rg_files(cmd, cwd=root, limit=limit)
        lines.sort()
        return lines

    return sorted(str(path.relative_to(root)) for path in root.glob(pattern))[:limit]


async def _run_rg_files(cmd: list[str], *, cwd: Path, limit: int) -> list[str]:
    session = get_sandbox()
    if session is not None and session.is_running:
        return await _run_rg_files_in_sandbox(cmd, cwd=cwd, limit=limit)

    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    return await _collect_limited_stdout(process, limit=limit)


async def _run_rg_files_in_sandbox(cmd: list[str], *, cwd: Path, limit: int) -> list[str]:
    import shlex

    session = get_sandbox()
    if session is None:
        return []
    shell_command = " ".join(shlex.quote(part) for part in cmd)
    output = await session.exec_command(f"cd {shlex.quote(str(cwd))} && {shell_command}")
    lines = [_normalize_rg_path(line.strip()) for line in output.splitlines() if line.strip()]
    return lines[:limit]


async def _collect_limited_stdout(
    process: asyncio.subprocess.Process,
    *,
    limit: int,
) -> list[str]:
    lines: list[str] = []

    async def read_stdout() -> None:
        assert process.stdout is not None
        while len(lines) < limit:
            raw = await process.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(_normalize_rg_path(line))

    try:
        try:
            await asyncio.wait_for(read_stdout(), timeout=_GLOB_RG_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            pass
    finally:
        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()

    return lines


def _normalize_rg_path(value: str) -> str:
    return value[2:] if value.startswith("./") else value
