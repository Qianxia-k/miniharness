"""Plugin system — discover, validate, and load extensions.

A plugin is a directory with a ``plugin.json`` manifest that contributes
skills, hooks, and MCP servers.  Plugins are discovered from user-level
and project-level directories, loaded, and integrated into the running
agent automatically.

Module map::

    schemas.py  — PluginManifest (Pydantic model)
    types.py    — LoadedPlugin (dataclass)
    loader.py   — discover + validate + load

Quick start::

    mkdir -p ~/.miniharness/plugins/my-plugin/skills/my-skill
    echo '{"name":"my-plugin"}' > ~/.miniharness/plugins/my-plugin/plugin.json
    echo '# my-skill' > ~/.miniharness/plugins/my-plugin/skills/my-skill/SKILL.md
"""

from miniharness.plugins.loader import (
    discover_plugin_paths,
    get_user_plugins_dir,
    load_plugins,
)
from miniharness.plugins.schemas import PluginManifest
from miniharness.plugins.types import LoadedPlugin

__all__ = [
    "LoadedPlugin",
    "PluginManifest",
    "discover_plugin_paths",
    "get_user_plugins_dir",
    "load_plugins",
]
