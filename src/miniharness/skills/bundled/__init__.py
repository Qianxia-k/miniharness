"""Bundled skills — shipped with MiniHarness.

These skills are loaded from the ``content/`` directory at package
installation time.  They are always available and have the lowest
priority (project and user skills can override them).
"""

from __future__ import annotations

from miniharness.skills.types import SkillDefinition


def get_bundled_skills() -> list[SkillDefinition]:
    """Load all bundled skills from ``content/*.md``.

    This is a separate function (not module-level) so import errors
    don't prevent the skills package from loading.
    """
    from miniharness.skills._frontmatter import parse_skill_frontmatter

    skills: list[SkillDefinition] = []
    try:
        import importlib.resources
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
        try:
            content = entry.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        default_name = entry.name[:-3]  # strip ".md"
        meta = parse_skill_frontmatter(
            content,
            default_name=default_name,
            fallback_template="Bundled skill: {name}",
        )
        skills.append(SkillDefinition(
            name=meta["name"],
            description=meta["description"],
            content=meta["body"],
            source="bundled",
            path=str(entry),
            base_dir=str(entry.parent),
        ))
    return skills
