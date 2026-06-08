from pathlib import Path

from miniharness.config.settings import Settings
from miniharness.plugins.loader import discover_plugin_paths, load_plugins


def _write_plugin(root: Path, name: str) -> Path:
    plugin_dir = root / name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.json").write_text(
        f'{{"name": "{name}", "description": "test plugin"}}',
        encoding="utf-8",
    )
    return plugin_dir


def test_project_plugins_are_not_discovered_by_default(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    user_plugins = home / ".miniharness" / "plugins"
    project_plugins = project / ".miniharness" / "plugins"
    user_plugins.mkdir(parents=True)
    project_plugins.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    user_plugin = _write_plugin(user_plugins, "user-plugin")
    _write_plugin(project_plugins, "project-plugin")

    paths = discover_plugin_paths(Settings(), cwd=project)

    assert paths == [user_plugin]


def test_project_plugins_require_explicit_trust(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    project = tmp_path / "project"
    project_plugins = project / ".miniharness" / "plugins"
    project_plugins.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))

    project_plugin = _write_plugin(project_plugins, "project-plugin")
    settings = Settings(allow_project_plugins=True)

    paths = discover_plugin_paths(settings, cwd=project)
    plugins = load_plugins(settings, cwd=project)

    assert paths == [project_plugin]
    assert [p.name for p in plugins] == ["project-plugin"]


def test_disabled_plugin_does_not_contribute_capabilities(tmp_path: Path, monkeypatch):
    home = tmp_path / "home"
    user_plugins = home / ".miniharness" / "plugins"
    plugin_dir = _write_plugin(user_plugins, "demo-plugin")
    skill_dir = plugin_dir / "skills" / "hello"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# hello\n\nHello skill.", encoding="utf-8")
    (plugin_dir / "mcp.json").write_text(
        '{"filesystem": {"type": "stdio", "command": "npx", "args": ["server"]}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    settings = Settings(enabled_plugins={"demo-plugin": False})
    plugins = load_plugins(settings, cwd=tmp_path)

    assert len(plugins) == 1
    assert plugins[0].enabled is False
    assert plugins[0].skills == []
    assert plugins[0].mcp_servers == {}
