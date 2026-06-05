"""Settings data model.

Mirrors OpenHarness's config/settings.py.  Each concern (provider, sandbox,
agent, hooks) owns its section so defaults, env vars, and overrides compose cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProviderSettings:
    """Resolved provider configuration."""

    name: str = "qwen"          # profile key: "qwen", "openai", ...
    model: str = ""               # empty = use profile's default_model
    base_url: str | None = None   # None = use profile's default base_url


@dataclass
class SandboxSettings:
    """Sandbox / container-isolation configuration."""

    enabled: bool = False
    image: str = "miniharness-sandbox:latest"
    fail_if_unavailable: bool = False


@dataclass
class AgentSettings:
    """LLM sampling parameters.

    All fields default to ``None``, meaning "use the provider's default".
    Only set values are forwarded to the API.
    """

    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


@dataclass
class HookSettings:
    """Hook system configuration.

    Controls which preset groups are active.  Individual hook overrides
    can be placed in ``custom_hooks`` (merged on top of presets).
    """

    # ── Preset groups (all enabled by default for production safety) ──
    dangerous_commands: bool = True
    """Block rm -rf, dd, mkfs, reboot, shutdown, fork bombs, etc."""

    sensitive_files: bool = True
    """Block writes to /etc, ~/.ssh, ~/.aws, /root, ~/.kube, etc."""

    code_security_review: bool = False
    """LLM-powered code security review before write_file/edit_file.
    DISABLED by default — costs extra API tokens.  Enable for
    high-security environments."""

    human_approval: bool = True
    """Require human 'yes' for: rm -rf, git push --force, kubectl delete,
    terraform destroy, DROP DATABASE, docker push, npm publish, etc."""

    audit_log: bool = True
    """Write JSONL audit trail to ~/.miniharness/audit/ (never blocks)."""

    audit_log_dir: str = "~/.miniharness/audit"
    """Directory for audit log files."""

    # ── Custom hooks (merged ON TOP of presets — can override) ──────
    custom_hooks: dict = field(default_factory=dict)
    """User-defined hooks.  Merged after presets; use to add extra hooks
    without disabling the built-in safety presets.

    Example::

        {
            "pre_tool_use": [
                {"type": "command", "command": "echo custom audit",
                 "matcher": "bash:*git*", "block_on_failure": False},
            ],
        }
    """


@dataclass
class Settings:
    """Top-level settings bag passed through the whole agent lifecycle.

    Every layer reads from this instead of reaching for env vars or CLI args
    directly — that way the loading chain is the single source of truth.
    """

    provider: ProviderSettings = field(default_factory=ProviderSettings)
    sandbox: SandboxSettings = field(default_factory=SandboxSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)
    hooks: HookSettings = field(default_factory=HookSettings)
    # {name: McpStdioServerConfig | McpHttpServerConfig | dict}
    mcp_servers: dict = field(default_factory=dict)
    # Plugin system — {plugin_name: True/False} for enabled/disabled.
    enabled_plugins: dict = field(default_factory=dict)
    allow_project_plugins: bool = True
    max_turns: int = 8
    context_budget_ratio: float = 0.8  # trigger compaction at 80% context usage
    keep_last_n_turns: int = 3  # when compacting, always keep the last N turns intact
