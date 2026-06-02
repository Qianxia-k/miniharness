"""Skill loader — discover and parse SKILL.md files from multiple sources.

Load order (later sources override earlier for the same skill name):

    1. Bundled skills (``skills/bundled/content/*.md``)
    2. Project skills (``.miniharness/skills/<name>/SKILL.md``)
    3. User skills (``~/.miniharness/skills/<name>/SKILL.md``)

Each source adds its skills to the registry; if a later source registers
a skill with the same name, it replaces the earlier one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniharness.skills._frontmatter import parse_bool, parse_skill_frontmatter
from miniharness.skills.registry import SkillRegistry
from miniharness.skills.types import SkillDefinition


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_skill_registry(
    cwd: str | Path | None = None,
    *,
    include_bundled: bool = True,
    include_project: bool = True,
    include_user: bool = True,
) -> SkillRegistry:
    """Load all skills into a single registry.

    Parameters
    ----------
    cwd:
        Project root for discovering project skills.  ``None`` = skip
        project discovery.
    include_bundled:
        Include skills bundled with MiniHarness.
    include_project:
        Discover skills from ``.miniharness/skills/`` directories
        walking up from *cwd*.
    include_user:
        Discover skills from ``~/.miniharness/skills/``.

    Returns
    -------
    SkillRegistry
        All discovered skills, with later sources overriding earlier ones.
    """
    registry = SkillRegistry()

    # 1. Bundled (lowest priority).
    if include_bundled:
        for skill in _load_bundled_skills():
            registry.register(skill)

    # 2. Project (medium priority).
    if include_project and cwd is not None:
        for skill in _load_project_skills(cwd):
            registry.register(skill)

    # 3. User (highest priority).
    if include_user:
        for skill in _load_user_skills():
            registry.register(skill)

    return registry


# ---------------------------------------------------------------------------
# Source: bundled
# ---------------------------------------------------------------------------


def _load_bundled_skills() -> list[SkillDefinition]:
    """Load skills bundled with MiniHarness from ``skills/bundled/content/``."""
    import importlib.resources

    skills: list[SkillDefinition] = []
    try:
        content_dir = importlib.resources.files(
            "miniharness.skills.bundled.content"
        )
    except (ModuleNotFoundError, TypeError):
        return skills

    if not content_dir.is_dir():
        return skills

    for entry in sorted(content_dir.iterdir()):
        if not entry.name.endswith(".md"):
            continue
        content = entry.read_text(encoding="utf-8")
        skill = _parse_skill_file(
            content=content,
            source="bundled",
            path=str(entry),
            base_dir=str(entry.parent),
        )
        if skill is not None:
            skills.append(skill)

    return skills


# ---------------------------------------------------------------------------
# Source: project
# ---------------------------------------------------------------------------


def _load_project_skills(cwd: str | Path) -> list[SkillDefinition]:
    """Discover skills from project directories walking up from *cwd*."""
    root = Path(cwd).resolve()
    # Walk up from cwd to find project skill dirs.
    # Stop at filesystem root or when we leave a git repo.
    dirs_to_check: list[Path] = []
    current = root
    while True:
        proj_dir = current / ".miniharness" / "skills"
        if proj_dir.is_dir():
            dirs_to_check.append(proj_dir)
        # Also check .claude/skills (compatibility).
        claude_dir = current / ".claude" / "skills"
        if claude_dir.is_dir():
            dirs_to_check.append(claude_dir)
        # Go up one level.
        parent = current.parent
        if parent == current or not str(parent).startswith("/"):
            break
        # Stop at git root boundary.
        if (current / ".git").is_dir() and not (parent / ".git").is_dir():
            # We're at the repo root — don't go above.
            pass
        current = parent

    # Process from outermost to innermost (innermost overrides).
    skills: list[SkillDefinition] = []
    for d in reversed(dirs_to_check):
        skills.extend(_load_skills_from_dir(d, source="project"))
    return skills


# ---------------------------------------------------------------------------
# Source: user
# ---------------------------------------------------------------------------


def _load_user_skills() -> list[SkillDefinition]:
    """Discover skills from ``~/.miniharness/skills/``."""
    user_dir = Path.home() / ".miniharness" / "skills"
    if not user_dir.is_dir():
        return []
    return _load_skills_from_dir(user_dir, source="user")


# ---------------------------------------------------------------------------
# Directory-based loading
# ---------------------------------------------------------------------------


def _load_skills_from_dir(
    root_dir: Path,
    source: str,
) -> list[SkillDefinition]:
    """Load skills from a directory tree: ``<root>/<name>/SKILL.md``.

    Each subdirectory of *root_dir* that contains a ``SKILL.md`` file
    produces one skill.
    """
    skills: list[SkillDefinition] = []
    if not root_dir.is_dir():
        return skills

    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue

        try:
            content = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        skill = _parse_skill_file(
            content=content,
            source=source,
            path=str(skill_file),
            base_dir=str(entry),
            default_name=entry.name,
        )
        if skill is not None:
            skills.append(skill)

    return skills


# ---------------------------------------------------------------------------
# Core parser — SKILL.md → SkillDefinition
# ---------------------------------------------------------------------------


def _parse_skill_file(
    *,
    content: str,
    source: str,
    path: str = "",
    base_dir: str = "",
    default_name: str = "unnamed",
) -> SkillDefinition | None:
    """Parse one SKILL.md file into a SkillDefinition.

    Returns ``None`` if the content is empty or unparseable.
    """
    if not content or not content.strip():
        return None

    fallback = _FALLBACK_TEMPLATES.get(source, "Skill: {name}")
    meta = parse_skill_frontmatter(
        content,
        default_name=default_name,
        fallback_template=fallback,
    )

    fm = meta.get("frontmatter", {})

    return SkillDefinition(
        name=meta["name"],
        description=meta["description"],
        content=meta["body"],
        source=source,
        path=path or None,
        base_dir=base_dir or None,
        model_invocable=not parse_bool(
            fm.get("disable_model_invocation", fm.get("disable-model-invocation")),
            default=False,
        ),
        user_invocable=parse_bool(
            fm.get("user_invocable", fm.get("user-invocable")),
            default=True,
        ),
    )


_FALLBACK_TEMPLATES: dict[str, str] = {
    "bundled": "Bundled skill: {name}",
    "project": "Project skill: {name}",
    "user": "User skill: {name}",
}
