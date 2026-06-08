"""Skill tool — lets the model load a skill by name at runtime.

When the model calls ``skill(name="code-review")``, this tool:
1. Looks up the skill in the registry.
2. Returns the full markdown content as instructions.
3. Records the invocation in ``tool_metadata["invoked_skills"]``.

Plugin skills use namespaced names such as ``demo-plugin:hello-world`` and are
blocked unless their plugin is active in the current runtime context.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from miniharness.tools.base import BaseTool, ToolResult
from miniharness.plugins.gating import is_plugin_active


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

    def __init__(
        self,
        *,
        cwd: Path,
        registry=None,
        permissions=None,
        plugin_index: list[dict] | None = None,
    ) -> None:
        super().__init__(cwd=cwd, permissions=permissions)
        self._registry = registry
        self._plugin_index = plugin_index or []

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

        plugin_name = getattr(skill, "plugin_name", None)
        if plugin_name and not is_plugin_active(plugin_name, self._plugin_index):
            return ToolResult(
                f"Plugin skill '{skill.invocation_name}' is not active in the current runtime context. "
                f"Activate plugin '{plugin_name}' first.",
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
            f"[Loaded skill: {skill.invocation_name}]\n\n{content}"
        )
