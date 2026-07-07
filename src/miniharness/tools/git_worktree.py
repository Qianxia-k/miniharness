"""Git worktree tools."""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolPermissionRequest, ToolResult
from miniharness.tools.git_utils import command_output, repo_root, run_git


class EnterWorktreeInput(BaseModel):
    """Arguments for enter_worktree."""

    branch: str = Field(description="Target branch name for the worktree")
    path: str | None = Field(default=None, description="Optional worktree path")
    create_branch: bool = Field(default=True, description="Create a new branch for the worktree")
    base_ref: str = Field(default="HEAD", description="Base ref when creating a new branch")


class ExitWorktreeInput(BaseModel):
    """Arguments for exit_worktree."""

    path: str = Field(description="Worktree path to remove")


class EnterWorktreeTool(BaseTool):
    """Create a git worktree and return its path."""

    name = "enter_worktree"
    description = "Create a git worktree for isolated coding work and return its path."
    input_model = EnterWorktreeInput

    async def execute(self, arguments: EnterWorktreeInput) -> ToolResult:
        clean_branch = arguments.branch.strip()
        if not clean_branch:
            return ToolResult("branch is required", is_error=True)

        root = repo_root(self.cwd)
        if root is None:
            return ToolResult("enter_worktree requires a git repository", is_error=True)

        worktree_path = _resolve_worktree_path(root, clean_branch, arguments.path)
        worktree_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = ["worktree", "add"]
        if arguments.create_branch:
            cmd.extend(["-b", clean_branch, str(worktree_path), arguments.base_ref])
        else:
            cmd.extend([str(worktree_path), clean_branch])

        result = run_git(root, *cmd, timeout=30)
        output = command_output(result)
        if result.returncode != 0:
            return ToolResult(output, is_error=True)
        return ToolResult(f"{output}\nPath: {worktree_path}")

    def permission_requests(self, arguments: EnterWorktreeInput) -> list[ToolPermissionRequest]:
        clean_branch = arguments.branch.strip()
        if not clean_branch:
            return []
        root = repo_root(self.cwd)
        if root is None:
            return []
        worktree_path = _resolve_worktree_path(root, clean_branch, arguments.path)
        command = _enter_worktree_command(arguments, worktree_path)
        return [ToolPermissionRequest(
            is_read_only=False,
            file_path=str(worktree_path),
            command=command,
            reason=f"Allow enter_worktree to create git worktree {worktree_path}?",
        )]


class ExitWorktreeTool(BaseTool):
    """Remove a git worktree by path."""

    name = "exit_worktree"
    description = "Remove a git worktree by path."
    input_model = ExitWorktreeInput

    async def execute(self, arguments: ExitWorktreeInput) -> ToolResult:
        raw_path = arguments.path.strip()
        if not raw_path:
            return ToolResult("path is required", is_error=True)

        path = _resolve_path(self.cwd, raw_path)
        result = run_git(self.cwd, "worktree", "remove", "--force", str(path), timeout=30)
        output = command_output(result)
        return ToolResult(output, is_error=result.returncode != 0)

    def permission_requests(self, arguments: ExitWorktreeInput) -> list[ToolPermissionRequest]:
        raw_path = arguments.path.strip()
        if not raw_path:
            return []
        path = _resolve_path(self.cwd, raw_path)
        return [ToolPermissionRequest(
            is_read_only=False,
            file_path=str(path),
            command=f"git worktree remove --force {path}",
            reason=f"Allow exit_worktree to remove git worktree {path}?",
        )]


def _resolve_worktree_path(repo_root: Path, branch: str, path: str | None) -> Path:
    if path:
        return _resolve_path(repo_root, path)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", branch).strip("-") or "worktree"
    return (repo_root / ".miniharness" / "worktrees" / slug).resolve()


def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _enter_worktree_command(arguments: EnterWorktreeInput, worktree_path: Path) -> str:
    branch = arguments.branch.strip()
    if arguments.create_branch:
        return f"git worktree add -b {branch} {worktree_path} {arguments.base_ref}"
    return f"git worktree add {worktree_path} {branch}"
