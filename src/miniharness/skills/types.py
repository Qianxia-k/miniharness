"""Skill definition — frozen dataclass representing one parsed SKILL.md.

A skill is a markdown file with optional YAML frontmatter that teaches
the model how to perform a specific task.  Skills can be:

- **bundled** with MiniHarness (in ``skills/bundled/content/``)
- **project** skills (in ``.miniharness/skills/<name>/SKILL.md``)
- **user** skills (in ``~/.miniharness/skills/<name>/SKILL.md``)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SkillDefinition:
    """One skill, parsed from a SKILL.md file.

    Attributes
    ----------
    name:
        Canonical name.  From YAML ``name`` field, or markdown ``# Heading``,
        or directory basename.
    description:
        Short description (one sentence).  From YAML ``description``, or
        first body paragraph, or fallback template.
    content:
        Full raw markdown body (everything after the YAML frontmatter block).
    source:
        Where this skill came from: ``"bundled"``, ``"project"``, or ``"user"``.
    path:
        Absolute filesystem path to the SKILL.md file (``None`` for
        programmatically-created skills).
    base_dir:
        Parent directory of SKILL.md.  Used for ``$SKILL_DIR`` template
        substitution when the skill references relative resources.
    model_invocable:
        Whether the model can load this skill via the ``skill`` tool.
        ``False`` = user-only (invoked via ``/<name>`` slash command).
    user_invocable:
        Whether users can invoke this skill directly via ``/<name>``.
    """

    name: str
    description: str
    content: str
    source: str = "project"
    path: str | None = None
    base_dir: str | None = None
    model_invocable: bool = True
    user_invocable: bool = True

    # ── Display helpers ────────────────────────────────────────────

    @property
    def command_name(self) -> str:
        """The name used for slash-commands (directory basename or skill name)."""
        if self.path:
            from pathlib import Path
            return Path(self.path).parent.name
        return self.name

    @property
    def display_name(self) -> str:
        """Human-readable name for listings."""
        cmd = self.command_name
        return self.name if self.name != cmd else cmd
