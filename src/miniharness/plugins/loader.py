"""Plugin loader — discover, validate, and load plugins from disk.

Plugin discovery order:
    1. User-level: ``~/.miniharness/plugins/<name>/plugin.json``
    2. Project-level: ``.miniharness/plugins/<name>/plugin.json``

Later sources override earlier ones (project overrides user for same name).

A plugin directory looks like::

    my-plugin/
      plugin.json         — manifest (required)
      skills/             — SKILL.md files (optional)
        code-review/SKILL.md
      hooks.json          — hook definitions (optional)
      mcp.json            — MCP server configs (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from miniharness.plugins.schemas import PluginManifest
from miniharness.plugins.types import LoadedPlugin


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_plugins(
    settings,
    *,
    cwd: str | Path | None = None,
) -> list[LoadedPlugin]:
    """Discover and load all plugins.

    Parameters
    ----------
    settings:
        ``Settings`` object for ``enabled_plugins`` overrides and
        ``allow_project_plugins`` control.
    cwd:
        Project root for discovering project-level plugins.

    Returns
    -------
    list[LoadedPlugin]
        All discovered plugins (both enabled and disabled).
    """
    plugin_paths = discover_plugin_paths(settings, cwd=cwd)
    enabled_map: dict[str, bool] = getattr(settings, "enabled_plugins", {}) or {}

    plugins: list[LoadedPlugin] = []
    for path in plugin_paths:
        plugin = _load_one(path, enabled_map)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def discover_plugin_paths(
    settings,
    *,
    cwd: str | Path | None = None,
) -> list[Path]:
    """Return sorted list of plugin directories to load.

    Project plugins are only included if ``allow_project_plugins`` is True.
    """
    roots: list[Path] = []
    seen: set[Path] = set()

    # 1. User-level: ~/.miniharness/plugins/
    user_dir = Path.home() / ".miniharness" / "plugins"
    roots.append(user_dir)

    # 2. Project-level: .miniharness/plugins/
    allow_project = getattr(settings, "allow_project_plugins", True)
    if allow_project and cwd is not None:
        proj_dir = Path(cwd).resolve() / ".miniharness" / "plugins"
        roots.append(proj_dir)

    # Scan each root: look for subdirectories containing plugin.json.
    result: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if entry in seen:
                continue
            manifest_path = _find_manifest(entry)
            if manifest_path is not None:
                seen.add(entry)
                result.append(entry)

    return result


def get_user_plugins_dir() -> Path:
    """Return (and create) the user plugins directory."""
    d = Path.home() / ".miniharness" / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Internal — single plugin loading
# ---------------------------------------------------------------------------


def _load_one(path: Path, enabled_map: dict[str, bool]) -> LoadedPlugin | None:
    """Load one plugin from *path*.

    Returns ``None`` if the manifest is missing or invalid.
    """
    manifest_path = _find_manifest(path)
    if manifest_path is None:
        return None

    try:
        raw = _parse_json_with_comments(manifest_path.read_text(encoding="utf-8"))
        manifest = PluginManifest(**raw)
    except Exception:
        return None

    enabled = enabled_map.get(manifest.name, manifest.enabled_by_default)

    # Load contributions.
    skills = _load_plugin_skills(path / manifest.skills_dir)
    hooks = _load_plugin_hooks(path / manifest.hooks_file)
    mcp_servers = _load_plugin_mcp(path / manifest.mcp_file)

    return LoadedPlugin(
        manifest=manifest,
        path=path,
        enabled=enabled,
        skills=skills,
        hooks=hooks,
        mcp_servers=mcp_servers,
    )


def _find_manifest(plugin_dir: Path) -> Path | None:
    """Find plugin.json in *plugin_dir*.

    Checks:
        1. ``plugin_dir / "plugin.json"``
        2. ``plugin_dir / ".miniharness-plugin" / "plugin.json"``
    """
    candidates = [
        plugin_dir / "plugin.json",
        plugin_dir / ".miniharness-plugin" / "plugin.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------
# Contribution loaders
# ---------------------------------------------------------------------------


def _load_plugin_skills(skills_dir: Path) -> list[Any]:
    """Load skills from a plugin's skills subdirectory.

    Uses the standard ``<name>/SKILL.md`` convention.
    """
    from miniharness.skills._frontmatter import parse_bool, parse_skill_frontmatter
    from miniharness.skills.types import SkillDefinition

    skills: list[SkillDefinition] = []
    if not skills_dir.is_dir():
        return skills

    for entry in sorted(skills_dir.iterdir()):
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue

        try:
            content = skill_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue

        meta = parse_skill_frontmatter(
            content,
            default_name=entry.name,
            fallback_template="Plugin skill: {name}",
        )
        fm = meta.get("frontmatter", {})
        skills.append(SkillDefinition(
            name=meta["name"],
            description=meta["description"],
            content=meta["body"],
            source="plugin",
            path=str(skill_file),
            base_dir=str(entry),
            model_invocable=not parse_bool(fm.get("disable_model_invocation"), default=False),
            user_invocable=parse_bool(fm.get("user_invocable", fm.get("user-invocable")), default=True),
        ))

    return skills


def _load_plugin_hooks(hooks_path: Path) -> dict[str, list[dict]]:
    """Load hook definitions from a plugin's hooks.json."""
    if not hooks_path.is_file():
        return {}

    try:
        raw = _parse_json_with_comments(hooks_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(raw, dict):
        return {}

    # Normalize: hooks.json can be {event: [hooks]} or {hooks: {event: [hooks]}}.
    hooks_dict = raw.get("hooks", raw)
    if not isinstance(hooks_dict, dict):
        return {}

    return {str(k): v for k, v in hooks_dict.items() if isinstance(v, list)}


def _load_plugin_mcp(mcp_path: Path) -> dict[str, Any]:
    """Load MCP server configs from a plugin's mcp.json."""
    if not mcp_path.is_file():
        return {}

    try:
        raw = _parse_json_with_comments(mcp_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    if not isinstance(raw, dict):
        return {}

    servers = raw.get("mcpServers", raw)
    return servers if isinstance(servers, dict) else {}


# ---------------------------------------------------------------------------
# JSON with comments support (shared with config loader)
# ---------------------------------------------------------------------------


def _parse_json_with_comments(text: str) -> dict:
    """Parse JSON that may contain ``//`` or ``#`` comment lines."""
    import re
    text = re.sub(r'^\s*//.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*#.*$', '', text, flags=re.MULTILINE)
    return json.loads(text)
