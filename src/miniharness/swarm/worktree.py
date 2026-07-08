"""Git worktree isolation for delegated agents."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

from miniharness.tools.git_utils import command_output, repo_root, run_git


_VALID_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_SLUG_LENGTH = 64


@dataclass(frozen=True)
class WorktreeInfo:
    """Metadata about a managed worktree."""

    slug: str
    path: Path
    branch: str
    original_path: Path
    created_at: float
    agent_id: str | None = None


def validate_worktree_slug(slug: str) -> str:
    """Validate a worktree slug and reject path traversal."""
    if not slug:
        raise ValueError("Worktree slug must not be empty")
    if len(slug) > _MAX_SLUG_LENGTH:
        raise ValueError(f"Worktree slug must be {_MAX_SLUG_LENGTH} characters or fewer")
    if slug.startswith("/") or slug.startswith("\\"):
        raise ValueError(f"Worktree slug must not be an absolute path: {slug!r}")
    for segment in slug.split("/"):
        if segment in ("", ".", ".."):
            raise ValueError(f"Worktree slug {slug!r} contains an invalid segment")
        if not _VALID_SEGMENT.match(segment):
            raise ValueError(
                "Worktree slug segments may contain only letters, digits, dots, "
                "underscores, and dashes"
            )
    return slug


class WorktreeManager:
    """Create and remove repo-local git worktrees for delegated agents."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir

    async def create_worktree(
        self,
        *,
        repo_path: Path,
        slug: str,
        branch: str | None = None,
        agent_id: str | None = None,
    ) -> WorktreeInfo:
        validate_worktree_slug(slug)
        root = repo_root(repo_path)
        if root is None:
            raise RuntimeError("worktree isolation requires a git repository")

        base_dir = self.base_dir or (root / ".miniharness" / "worktrees")
        base_dir.mkdir(parents=True, exist_ok=True)

        flat_slug = slug.replace("/", "+")
        worktree_path = (base_dir / flat_slug).resolve()
        worktree_branch = branch or f"worktree-{flat_slug}"

        if worktree_path.exists():
            existing = run_git(worktree_path, "rev-parse", "--git-dir")
            if existing.returncode == 0:
                return WorktreeInfo(
                    slug=slug,
                    path=worktree_path,
                    branch=worktree_branch,
                    original_path=root,
                    created_at=worktree_path.stat().st_mtime,
                    agent_id=agent_id,
                )

        result = run_git(
            root,
            "worktree",
            "add",
            "-B",
            worktree_branch,
            str(worktree_path),
            "HEAD",
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {command_output(result)}")

        return WorktreeInfo(
            slug=slug,
            path=worktree_path,
            branch=worktree_branch,
            original_path=root,
            created_at=time.time(),
            agent_id=agent_id,
        )

    async def remove_worktree(self, path_or_slug: str | Path) -> bool:
        """Remove a worktree by absolute path or manager-local slug."""
        path = Path(path_or_slug).expanduser()
        if not path.is_absolute():
            if self.base_dir is None:
                raise ValueError("relative worktree removal requires a manager base_dir")
            path = self.base_dir / path
        path = path.resolve()
        if not path.exists():
            return False

        common_dir = run_git(path, "rev-parse", "--git-common-dir")
        if common_dir.returncode == 0 and common_dir.stdout.strip():
            repo = Path(common_dir.stdout.strip()).resolve().parent
            result = run_git(repo, "worktree", "remove", "--force", str(path), timeout=30)
            return result.returncode == 0

        result = run_git(path.parent, "worktree", "remove", "--force", str(path), timeout=30)
        return result.returncode == 0


def worktree_slug_for_agent(*, agent_id: str, description: str = "") -> str:
    """Build a deterministic slug for one delegated agent."""
    base = agent_id.strip() or description.strip() or "agent"
    slug = re.sub(r"[^A-Za-z0-9._/-]+", "-", base).strip("-/") or "agent"
    slug = slug.replace("@", "/")
    if len(slug) <= _MAX_SLUG_LENGTH:
        return slug
    return slug[:_MAX_SLUG_LENGTH].rstrip("-/.") or "agent"
