"""Coordinator/delegation prompt context for MiniHarness."""

from __future__ import annotations

from pathlib import Path

from miniharness.coordinator.agent_definitions import get_all_agent_definitions_for_plugins


def build_delegation_section(
    cwd: str | Path,
    *,
    plugin_index: list[dict] | None = None,
) -> str | None:
    """Build the delegated-agent guidance block for the model.

    This mirrors OpenHarness's coordinator guidance in spirit: MiniHarness is
    the coordinator, workers are asynchronous background agents, and worker
    prompts must be self-contained because workers do not see the parent
    conversation.
    """
    try:
        plugins = _plugins_from_index(plugin_index)
        agents = get_all_agent_definitions_for_plugins(cwd=cwd, plugins=plugins)
    except Exception:
        agents = []
    if not agents:
        return None

    lines = [
        "# Delegation And Subagents",
        "",
        "MiniHarness can delegate focused background work with the `agent` tool.",
        "Use it when the user explicitly asks for a subagent/background worker, "
        "when independent work can safely proceed in parallel, or when a focused "
        "worker materially improves the result.",
        "",
        "## Coordinator Role",
        "",
        "You are the coordinator. Answer directly when possible, and delegate only "
        "when delegation helps. Every normal assistant message you send is to the "
        "user. Background task notifications are internal signals; summarize their "
        "useful information for the user instead of treating them as conversation partners.",
        "",
        "## Default Pattern",
        "",
        '- Spawn implementation work with `agent(description=..., prompt=..., subagent_type="worker")`.',
        '- Spawn verification work with `agent(description=..., prompt=..., subagent_type="verification")`.',
        "- Inspect running or recorded workers with `/agents`, `agent_list`, and task tools.",
        "- Read worker output with `task_output(task_id=...)`.",
        "- Continue a worker with `send_message(task_id=..., message=...)` when its existing context helps.",
        "",
        "After launching workers, briefly tell the user what you launched and stop. "
        "Do not invent worker results; results arrive later through task output or notifications.",
        "",
        "## Worker Prompt Rules",
        "",
        "Workers cannot see your parent conversation. Every worker prompt must be self-contained:",
        "- include the concrete goal and relevant user requirements;",
        "- include file paths, line numbers, command outputs, or errors the worker needs;",
        "- state whether the worker may modify files or must stay read-only;",
        "- state what done looks like and what to report back.",
        "",
        "Never write vague prompts such as 'based on your findings' or 'fix the bug we discussed'. "
        "Synthesize the findings yourself and give the worker a specific task.",
        "",
        "## Concurrency And Follow-up",
        "",
        "- Run independent read-only research workers in parallel when useful.",
        "- Avoid concurrent write-heavy workers touching the same files.",
        "- Continue the same worker for corrections, follow-up work on the same files, or failures where its context helps.",
        "- Spawn a fresh verification worker to review implementation with less bias.",
        "- Spawn fresh when the old worker explored broadly but the next task is narrow.",
        "",
        "## Verification Standard",
        "",
        "Verification means proving behavior works, not confirming files exist. "
        "Ask verification workers to run commands, inspect outputs, and try at least one edge case. "
        "They should report PASS, FAIL, or PARTIAL with evidence.",
        "",
        "## Available Delegated Agent Definitions",
        "",
    ]
    for agent in agents:
        model = f" model={agent.model}" if agent.model else ""
        lines.append(f"- **{agent.name}**{model}: {agent.description}")

    lines.extend([
        "",
        "Prefer a normal direct answer for simple tasks. Use subagents only when they materially help.",
    ])
    return "\n".join(lines)


def _plugins_from_index(plugin_index: list[dict] | None) -> list | None:
    if plugin_index is None:
        return None
    plugins = []
    for entry in plugin_index:
        plugin = entry.get("_plugin") if isinstance(entry, dict) else None
        if plugin is not None:
            plugins.append(plugin)
    return plugins
