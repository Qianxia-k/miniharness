"""MCP configuration — load and validate server configs from settings."""

from __future__ import annotations

from pathlib import Path

from miniharness.mcp.types import (
    McpHttpServerConfig,
    McpServerConfig,
    McpStdioServerConfig,
)


def load_mcp_server_configs(
    settings,
    *,
    cwd: str | Path | None = None,
) -> dict[str, McpServerConfig]:
    """Extract and validate MCP server configs from ``Settings.mcp_servers``.

    Handles both typed configs (``McpStdioServerConfig``, ``McpHttpServerConfig``)
    and raw dicts that look like valid configs.

    Parameters
    ----------
    settings:
        A ``Settings`` object with an ``mcp_servers`` field (dict).

    Returns
    -------
    dict[str, McpServerConfig]
        Validated server configs ready for ``McpClientManager``.
    """
    raw = getattr(settings, "mcp_servers", None)
    if raw is None or not isinstance(raw, dict):
        return {}
    workspace = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()

    configs: dict[str, McpServerConfig] = {}
    for name, cfg in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue

        # Already a typed config → use as-is.
        if isinstance(cfg, McpStdioServerConfig):
            configs[name] = _normalize_stdio_config(cfg, workspace=workspace)
            continue
        if isinstance(cfg, McpHttpServerConfig):
            configs[name] = cfg
            continue

        # Raw dict → try to parse as typed config.
        if isinstance(cfg, dict):
            cfg_type = cfg.get("type", "")
            try:
                if cfg_type == "stdio":
                    # Support both "command" + "args" and flat strings.
                    normalized = dict(cfg)
                    if "args" not in normalized:
                        normalized["args"] = []
                    if "command" not in normalized:
                        continue  # stdio requires a command.
                    configs[name] = _normalize_stdio_config(
                        McpStdioServerConfig(**normalized),
                        workspace=workspace,
                    )
                elif cfg_type == "http":
                    if "url" not in cfg:
                        continue
                    configs[name] = McpHttpServerConfig(**cfg)
                else:
                    continue  # unknown type → skip.
            except Exception:
                continue  # invalid config → skip.

    return configs


def _normalize_stdio_config(
    config: McpStdioServerConfig,
    *,
    workspace: Path,
) -> McpStdioServerConfig:
    """Normalize stdio MCP config before launching a subprocess.

    Filesystem MCP servers accept allowed directories as trailing CLI args.
    MiniHarness exposes those directories as first-class config fields so the
    security boundary is visible in settings instead of being hidden inside a
    command-line list.
    """
    allowed = _normalized_allowed_directories(config, workspace=workspace)
    if not allowed or not _looks_like_filesystem_server(config):
        return config

    args = list(config.args)
    for directory in allowed:
        if directory not in args:
            args.append(directory)
    return config.model_copy(update={"args": args, "allowed_directories": allowed})


def _normalized_allowed_directories(
    config: McpStdioServerConfig,
    *,
    workspace: Path,
) -> list[str]:
    values = [*config.allowed_directories, *config.roots]
    seen: set[str] = set()
    result: list[str] = []
    base = _expand_path_template(config.cwd, workspace=workspace) if config.cwd else workspace
    for raw in values:
        if not isinstance(raw, str) or not raw.strip():
            continue
        path = _expand_path_template(raw, workspace=workspace)
        if not path.is_absolute():
            path = base / path
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return result


def _expand_path_template(value: str, *, workspace: Path) -> Path:
    home = Path.home().resolve()
    expanded = (
        value.replace("${cwd}", str(workspace))
        .replace("${workspace}", str(workspace))
        .replace("${project}", str(workspace))
        .replace("${home}", str(home))
    )
    return Path(expanded).expanduser()


def _looks_like_filesystem_server(config: McpStdioServerConfig) -> bool:
    haystack = " ".join([config.command, *config.args]).lower()
    return "filesystem" in haystack and (
        "mcp" in haystack or "modelcontextprotocol" in haystack
    )
