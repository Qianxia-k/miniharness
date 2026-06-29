"""Agent definition loading for delegated MiniHarness agents.

Agent definitions turn ``subagent_type`` from a cosmetic label into a stable
role contract.  Definitions may be built in or loaded from markdown files in:

- ``~/.miniharness/agents/*.md``
- ``<project>/.miniharness/agents/*.md``

Markdown body becomes the role system prompt.  Optional YAML frontmatter can
set ``name``, ``description``, ``model``, ``system_prompt_mode`` and future
policy metadata such as ``tools`` or ``permission_mode``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from miniharness.skills._frontmatter import parse_bool, parse_skill_frontmatter


AgentSource = Literal["builtin", "user", "project", "plugin"]
SystemPromptMode = Literal["append", "replace"]


@dataclass(frozen=True)
class AgentDefinition:
    """Configuration for a named delegated agent role."""

    name: str
    description: str
    system_prompt: str = ""
    model: str | None = None
    system_prompt_mode: SystemPromptMode = "append"
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=list)
    permission_mode: str | None = None
    max_turns: int | None = None
    hooks: dict[str, Any] | None = None
    background: bool = False
    initial_prompt: str | None = None
    subagent_type: str | None = None
    source: AgentSource = "builtin"
    path: str | None = None

    def spawn_name(self) -> str:
        return (self.subagent_type or self.name).strip() or "agent"


_GENERAL_PURPOSE_PROMPT = """You are a delegated MiniHarness agent.

Work independently on the assigned task. Use available tools when needed,
follow the project conventions, and return a concise report with what you did,
what you found, and any remaining risks.
"""

_EXPLORE_PROMPT = """You are a read-only codebase exploration agent.

Your job is to search, inspect, and explain existing code. Do not create,
modify, move, or delete files. Use read-only tools and read-only shell commands
only. Return concise findings with relevant file paths and evidence.
"""

_PLAN_PROMPT = """You are a planning agent.

Explore the codebase and produce an implementation plan. Do not modify files.
Identify the critical files, risks, sequencing, and verification strategy.
End with a compact checklist the parent agent can execute.
"""

_WORKER_PROMPT = """You are an implementation-focused worker agent.

Make concrete code changes for the assigned task. Keep changes scoped, follow
the repository style, and run relevant verification before reporting back.
Return the files changed, tests run, and any limitations.
"""

_VERIFICATION_PROMPT = """You are a verification-only agent.

Your job is to test whether the implementation actually works. Do not modify
project files. Run relevant commands, inspect outputs, and try at least one
edge or adversarial case. End with exactly one verdict line:

VERDICT: PASS
VERDICT: FAIL
VERDICT: PARTIAL
"""


_BUILTIN_AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        name="general-purpose",
        description="General delegated agent for research, code search, and multi-step tasks.",
        system_prompt=_GENERAL_PURPOSE_PROMPT,
        subagent_type="general-purpose",
    ),
    AgentDefinition(
        name="Explore",
        description="Read-only code exploration agent for searching and explaining codebases.",
        system_prompt=_EXPLORE_PROMPT,
        model="inherit",
        disallowed_tools=["write_file", "edit_file", "agent"],
        permission_mode="plan",
        subagent_type="Explore",
    ),
    AgentDefinition(
        name="Plan",
        description="Read-only planning agent for implementation strategy and architecture.",
        system_prompt=_PLAN_PROMPT,
        model="inherit",
        disallowed_tools=["write_file", "edit_file", "agent"],
        permission_mode="plan",
        subagent_type="Plan",
    ),
    AgentDefinition(
        name="worker",
        description="Implementation-focused worker for coding tasks and test execution.",
        system_prompt=_WORKER_PROMPT,
        subagent_type="worker",
    ),
    AgentDefinition(
        name="verification",
        description="Verification-only agent for testing implementation work before completion.",
        system_prompt=_VERIFICATION_PROMPT,
        model="inherit",
        disallowed_tools=["write_file", "edit_file", "agent"],
        permission_mode="plan",
        subagent_type="verification",
    ),
)


def get_builtin_agent_definitions() -> list[AgentDefinition]:
    """Return built-in agent definitions."""
    return list(_BUILTIN_AGENTS)


def load_agents_dir(directory: Path, *, source: AgentSource = "user") -> list[AgentDefinition]:
    """Load markdown agent definitions from one directory."""
    if not directory.is_dir():
        return []

    definitions: list[AgentDefinition] = []
    for path in sorted(directory.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        meta = parse_skill_frontmatter(
            content,
            default_name=path.stem,
            fallback_template="Agent: {name}",
        )
        fm = meta.get("frontmatter", {})
        if not isinstance(fm, dict):
            fm = {}

        name = _string(fm.get("name")) or str(meta.get("name") or path.stem).strip()
        description = _string(fm.get("description")) or str(meta.get("description") or "").strip()
        body = str(meta.get("body") or "").strip()
        if not name or not description:
            continue

        mode = _string(fm.get("system_prompt_mode")) or _string(
            fm.get("systemPromptMode")
        )
        definitions.append(
            AgentDefinition(
                name=name,
                description=description,
                system_prompt=_string(fm.get("system_prompt")) or body,
                model=_normalize_model(_string(fm.get("model"))),
                system_prompt_mode="replace" if mode == "replace" else "append",
                tools=_parse_str_list(fm.get("tools")),
                disallowed_tools=_parse_str_list(
                    fm.get("disallowed_tools", fm.get("disallowedTools"))
                ) or [],
                permission_mode=_string(
                    fm.get("permission_mode", fm.get("permissionMode"))
                ) or None,
                max_turns=_parse_positive_int(
                    fm.get("max_turns", fm.get("maxTurns"))
                ),
                hooks=fm.get("hooks") if isinstance(fm.get("hooks"), dict) else None,
                background=parse_bool(fm.get("background"), default=False),
                initial_prompt=_string(
                    fm.get("initial_prompt", fm.get("initialPrompt"))
                ) or None,
                subagent_type=_string(fm.get("subagent_type")) or name,
                source=source,
                path=str(path),
            )
        )
    return definitions


def get_all_agent_definitions(*, cwd: str | Path | None = None) -> list[AgentDefinition]:
    """Return built-in, user, and project agent definitions with override order."""
    return _get_all_agent_definitions(cwd=cwd, plugins=None)


def get_all_agent_definitions_for_plugins(
    *,
    cwd: str | Path | None = None,
    plugins: list | None = None,
) -> list[AgentDefinition]:
    """Return agent definitions including already-loaded plugin contributions."""
    return _get_all_agent_definitions(cwd=cwd, plugins=plugins)


def _get_all_agent_definitions(
    *,
    cwd: str | Path | None = None,
    plugins: list | None = None,
) -> list[AgentDefinition]:
    agent_map: dict[str, AgentDefinition] = {}
    for definition in get_builtin_agent_definitions():
        agent_map[definition.name] = definition

    for definition in load_agents_dir(Path.home() / ".miniharness" / "agents", source="user"):
        agent_map[definition.name] = definition

    if cwd is not None:
        project_dir = Path(cwd).expanduser().resolve() / ".miniharness" / "agents"
        for definition in load_agents_dir(project_dir, source="project"):
            agent_map[definition.name] = definition

    if plugins is not None:
        for plugin in plugins:
            if not getattr(plugin, "enabled", False):
                continue
            for definition in getattr(plugin, "agents", []) or []:
                if isinstance(definition, AgentDefinition):
                    agent_map[definition.name] = definition
    else:
        try:
            from miniharness.config import load_settings
            from miniharness.plugins.loader import load_plugins

            settings = load_settings()
            for plugin in load_plugins(settings, cwd=cwd):
                if not plugin.enabled:
                    continue
                for definition in plugin.agents:
                    if isinstance(definition, AgentDefinition):
                        agent_map[definition.name] = definition
        except Exception:
            pass

    return sorted(agent_map.values(), key=lambda item: item.name.lower())


def get_agent_definition(
    name: str | None,
    *,
    cwd: str | Path | None = None,
    plugins: list | None = None,
) -> AgentDefinition | None:
    """Return a named agent definition, case-insensitively."""
    requested = (name or "").strip()
    if not requested:
        return None
    lowered = requested.lower()
    for definition in _get_all_agent_definitions(cwd=cwd, plugins=plugins):
        aliases = {definition.name.lower(), definition.spawn_name().lower()}
        if lowered in aliases:
            return definition
    return None


def merge_agent_definition(
    definition: AgentDefinition | None,
    *,
    fallback_name: str,
    fallback_description: str,
) -> AgentDefinition:
    """Return a concrete definition for spawn-time use."""
    if definition is None:
        return AgentDefinition(
            name=fallback_name,
            description=fallback_description,
            subagent_type=fallback_name,
            source="builtin",
        )
    if definition.description:
        return definition
    return replace(definition, description=fallback_description)


def _string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _normalize_model(value: str) -> str | None:
    if not value:
        return None
    return value


def _parse_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.replace("\n", ",").split(",")]
        return [item for item in items if item]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return None


def _parse_positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
