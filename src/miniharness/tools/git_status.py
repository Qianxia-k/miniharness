"""Read-only git repository status tool."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult
from miniharness.tools.git_utils import command_output, git_text, repo_root, run_git


class GitStatusInput(BaseModel):
    """Arguments for git_status."""

    include_diff_stat: bool = Field(
        default=True,
        description="Include short --stat summaries for staged and unstaged diffs",
    )
    include_untracked: bool = Field(
        default=True,
        description="Include untracked files in the porcelain status output",
    )
    max_entries: int = Field(
        default=80,
        ge=1,
        le=500,
        description="Maximum number of status entries to return",
    )


class GitStatusTool(BaseTool):
    """Inspect the current git repository without mutating it."""

    name = "git_status"
    description = (
        "Show read-only git repository status: root, branch, HEAD, upstream, "
        "dirty files, and optional diff stats."
    )
    input_model = GitStatusInput

    def is_read_only(self, arguments: GitStatusInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: GitStatusInput) -> ToolResult:
        repo_root = repo_root_for_status(self.cwd)
        if repo_root is None:
            return ToolResult("git_status requires a git repository", is_error=True)

        branch = git_text(repo_root, "branch", "--show-current") or "(detached)"
        head = git_text(repo_root, "rev-parse", "--short", "HEAD") or "unknown"
        upstream = git_text(repo_root, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
        ahead_behind = _ahead_behind(repo_root) if upstream else ""

        status_args = ["status", "--porcelain=v1"]
        if not arguments.include_untracked:
            status_args.append("--untracked-files=no")
        status = run_git(repo_root, *status_args)
        if status.returncode != 0:
            return ToolResult(command_output(status), is_error=True)
        entries = [line for line in status.stdout.splitlines() if line.strip()]
        shown = entries[: arguments.max_entries]

        lines = [
            f"Repository: {repo_root}",
            f"Branch: {branch}",
            f"HEAD: {head}",
            f"Upstream: {upstream or '(none)'}{ahead_behind}",
            f"Status: {'clean' if not entries else f'{len(entries)} changed entrie(s)'}",
        ]

        if shown:
            lines.append("")
            lines.append("Changed files:")
            lines.extend(f"  {entry}" for entry in shown)
            omitted = len(entries) - len(shown)
            if omitted > 0:
                lines.append(f"  ... {omitted} more entrie(s) omitted")

        if arguments.include_diff_stat:
            staged = git_text(repo_root, "diff", "--cached", "--stat")
            unstaged = git_text(repo_root, "diff", "--stat")
            if staged:
                lines.append("")
                lines.append("Staged diff stat:")
                lines.append(staged)
            if unstaged:
                lines.append("")
                lines.append("Unstaged diff stat:")
                lines.append(unstaged)

        return ToolResult("\n".join(lines))


def repo_root_for_status(cwd: Path) -> Path | None:
    return repo_root(cwd)


def _ahead_behind(repo_root: Path) -> str:
    result = run_git(repo_root, "rev-list", "--left-right", "--count", "HEAD...@{u}")
    if result.returncode != 0:
        return ""
    parts = result.stdout.strip().split()
    if len(parts) != 2:
        return ""
    ahead, behind = parts
    return f" (ahead {ahead}, behind {behind})"
