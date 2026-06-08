"""Skill registry — in-memory index of all loaded skills.

Each skill is indexed under multiple keys (name, command_name, aliases)
so lookups are fast and case-insensitive.
"""

from __future__ import annotations

from miniharness.skills.types import SkillDefinition


class SkillRegistry:
    """Thread-safe-ish registry mapping skill keys → SkillDefinition.

    Usage::

        registry = SkillRegistry()
        registry.register(skill)
        found = registry.get("code-review")
        all_skills = registry.list_skills()
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill under all its lookup keys.

        Direct skills are indexed by name/command/display keys. Plugin skills
        are indexed only by their namespaced invocation key
        (``plugin-name:skill-name``), preventing silent collisions with direct
        skills or other plugins.
        Later registrations for the same key overwrite earlier ones
        (project skills can override bundled, user skills override project).
        """
        for key in self._keys_for(skill):
            self._skills[key] = skill

    def get(self, name: str) -> SkillDefinition | None:
        """Look up a skill by name.

        Tries exact match, then lowercase, then title case.
        """
        skill = self._skills.get(name)
        if skill is not None:
            return skill
        skill = self._skills.get(name.lower())
        if skill is not None:
            return skill
        return self._skills.get(name.title())

    def list_skills(self) -> list[SkillDefinition]:
        """Return all unique skills, sorted by name."""
        # Deduplicate by (source, path) — same skill may be registered
        # under multiple keys.
        seen: dict[tuple[str, str, str], SkillDefinition] = {}
        for skill in self._skills.values():
            key = (skill.source, skill.plugin_name or "", skill.path or skill.name)
            seen[key] = skill
        return sorted(seen.values(), key=lambda s: s.name)

    def model_invocable_skills(self) -> list[SkillDefinition]:
        """Return skills the model is allowed to load via the skill tool."""
        return [s for s in self.list_skills() if s.model_invocable]

    @property
    def count(self) -> int:
        """Total unique skills."""
        return len(self.list_skills())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _keys_for(skill: SkillDefinition) -> list[str]:
        """Build lookup keys for a skill."""
        keys: list[str] = []
        if skill.source == "plugin" or skill.plugin_name:
            invocation = skill.invocation_name
            return [invocation, invocation.lower()]

        # Canonical name.
        if skill.name:
            keys.append(skill.name)
            keys.append(skill.name.lower())
        # Command name (directory basename).
        cmd = skill.command_name
        if cmd and cmd != skill.name:
            keys.append(cmd)
            keys.append(cmd.lower())
        # Display name.
        if skill.display_name and skill.display_name != skill.name:
            keys.append(skill.display_name)
        return keys
