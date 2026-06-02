"""Skills system — markdown-based agent capabilities.

A skill is a ``SKILL.md`` file that teaches the model how to perform a
specific task.  Skills are:

- **Discovered** from bundled, project, and user directories.
- **Listed** in the system prompt so the model knows what's available.
- **Loaded** at runtime via the ``skill(name="...")`` tool.
- **Invoked** by users via ``/<name>`` slash commands (future).

Module map::

    types.py         — SkillDefinition frozen dataclass
    _frontmatter.py  — YAML frontmatter parser
    registry.py      — SkillRegistry (lookup, listing)
    loader.py        — Directory discovery + loading
    tool.py          — SkillTool (model-invocable)
    bundled/         — Built-in skills (commit, review, test)
"""

from miniharness.skills.loader import load_skill_registry
from miniharness.skills.registry import SkillRegistry
from miniharness.skills.tool import SkillTool
from miniharness.skills.types import SkillDefinition

__all__ = [
    "SkillDefinition",
    "SkillRegistry",
    "SkillTool",
    "load_skill_registry",
]
