"""Read-only git diff tool and command helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult
from miniharness.tools.git_utils import command_output, repo_root, run_git


GitDiffScope = Literal["unstaged", "staged", "head"]


class GitDiffInput(BaseModel):
    """Arguments for git_diff."""

    scope: GitDiffScope = Field(
        default="unstaged",
        description=(
            "Diff scope: 'unstaged' for working tree changes, 'staged' for "
            "index changes, or 'head' for all changes relative to HEAD"
        ),
    )
    stat_only: bool = Field(default=False, description="Return --stat summary instead of full diff")
    path: str | None = Field(default=None, description="Optional pathspec to limit the diff")
    max_chars: int = Field(default=20000, ge=1000, le=100000, description="Maximum diff characters to return")


class GitDiffTool(BaseTool):
    """Show git diff output without mutating the repository."""

    name = "git_diff"
    description = "Show read-only git diff output for unstaged, staged, or HEAD changes."
    input_model = GitDiffInput

    def is_read_only(self, arguments: GitDiffInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: GitDiffInput) -> ToolResult:
        rendered = render_git_diff(
            self.cwd,
            scope=arguments.scope,
            stat_only=arguments.stat_only,
            path=arguments.path,
            max_chars=arguments.max_chars,
        )
        return ToolResult(rendered.output, is_error=rendered.is_error)


class RenderedGitDiff(BaseModel):
    """Rendered git diff command result."""

    output: str
    is_error: bool = False


def render_git_diff(
    cwd: Path,
    *,
    scope: GitDiffScope = "unstaged",
    stat_only: bool = False,
    path: str | None = None,
    max_chars: int = 20000,
) -> RenderedGitDiff:
    """Render a bounded git diff from one shared implementation."""
    root = repo_root(cwd)
    if root is None:
        return RenderedGitDiff(output="git_diff requires a git repository", is_error=True)

    args = _diff_args(scope=scope, stat_only=stat_only)
    clean_path = (path or "").strip()
    if clean_path:
        args.extend(["--", clean_path])

    result = run_git(root, *args, timeout=20)
    output = command_output(result)
    if result.returncode != 0:
        return RenderedGitDiff(output=output, is_error=True)
    if not output:
        return RenderedGitDiff(output="(no diff)")
    return RenderedGitDiff(output=_truncate_diff(output, max_chars=max_chars))


def render_diff_command(cwd: Path, args: str) -> RenderedGitDiff:
    """Parse `/diff` arguments and render a diff for slash commands.

    Matches the common CLI ergonomics:
    - `/diff` -> unstaged diff stat
    - `/diff full` -> full diff against HEAD
    - `/diff staged` -> full staged diff
    - `/diff head` -> full HEAD diff
    - `/diff stat` -> unstaged diff stat
    - `/diff <path>` -> unstaged full diff for a path
    """
    raw = args.strip()
    if not raw or raw == "stat":
        return render_git_diff(cwd, scope="unstaged", stat_only=True)
    if raw == "full":
        return render_git_diff(cwd, scope="head", stat_only=False)
    if raw == "staged":
        return render_git_diff(cwd, scope="staged", stat_only=False)
    if raw == "head":
        return render_git_diff(cwd, scope="head", stat_only=False)
    if raw.startswith("stat "):
        return render_git_diff(cwd, scope="unstaged", stat_only=True, path=raw[5:].strip())
    return render_git_diff(cwd, scope="unstaged", stat_only=False, path=raw)


def _diff_args(*, scope: GitDiffScope, stat_only: bool) -> list[str]:
    args = ["diff"]
    if scope == "staged":
        args.append("--cached")
    elif scope == "head":
        args.append("HEAD")
    if stat_only:
        args.append("--stat")
    return args


def _truncate_diff(output: str, *, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    omitted = len(output) - max_chars
    return output[:max_chars] + f"\n... ({omitted} diff chars omitted)"
