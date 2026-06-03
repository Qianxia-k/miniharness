"""Skill tool — lets the model load a skill by name at runtime.

When the model calls ``skill(name="code-review")``, this tool:
1. Looks up the skill in the registry.
2. Returns the full markdown content as instructions.
3. Records the invocation in ``tool_metadata["invoked_skills"]``.

The skill registry is reloaded on each invocation so dynamically-added
skills (project/user) are always visible.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult


class SkillToolInput(BaseModel):
    """Arguments for the skill tool."""

    name: str = Field(description="The name of the skill to load")


class SkillTool(BaseTool):
    """Load a bundled, project, or user skill by name.

    The model uses this tool to fetch detailed instructions before
    performing a task that matches a skill's description.
    """

    name = "skill"
    description = (
        "Load a skill by name to get detailed instructions for a specific "
        "task.  Use this when a user's request matches a skill description "
        "listed in the Available Skills section of the system prompt."
    )
    input_model = SkillToolInput

    def __init__(self, *, cwd: Path, registry=None, permissions=None) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._registry = registry

    async def execute(self, arguments: SkillToolInput) -> ToolResult:
        """Load and return the skill's content.

        The skill registry is reloaded from disk on each invocation to
        pick up dynamically-added project/user skills.
        """
        if self._registry is None:
            return ToolResult("Skill registry not available.", is_error=True)

        skill_name = arguments.name.strip()
        skill = self._registry.get(skill_name)
        if skill is None:
            return ToolResult(
                f"Skill not found: '{skill_name}'. "
                f"Check available skills with the system prompt or try a different name.",
                is_error=True,
            )

        if not skill.model_invocable:
            return ToolResult(
                f"Skill '{skill.name}' can only be invoked by the user, "
                f"not by the model.",
                is_error=True,
            )

        # Apply template substitutions.
        content = skill.content
        if skill.base_dir:
            content = content.replace("${SKILL_DIR}", skill.base_dir)
            # Also support absolute path for bash commands.
            content = (
                f"Base directory for this skill: {skill.base_dir}\n\n"
                + content
            )

        return ToolResult(
            f"[Loaded skill: {skill.name}]\n\n{content}"
        )
