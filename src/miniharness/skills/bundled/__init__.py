"""Bundled skills — shipped with MiniHarness.

Directory layout (standard ``<name>/SKILL.md`` convention)::

    bundled/content/
      commit/SKILL.md
      code-review/SKILL.md
      test/SKILL.md

Each subdirectory is one skill.  Bundled skills have the lowest priority —
project and user skills with the same name override them.
"""

from __future__ import annotations

from miniharness.skills.types import SkillDefinition


def get_bundled_skills() -> list[SkillDefinition]:
    """Load all bundled skills from ``content/<name>/SKILL.md``."""
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
        # Each bundled skill is a subdirectory containing SKILL.md.
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue

        try:
            content = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        # Default name from the subdirectory name (e.g. "code-review").
        default_name = entry.name

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
            path=str(skill_file),
            base_dir=str(entry),  # skill-specific directory
        ))

    return skills
