"""Built-in slash commands — always available.

Each function is a ``CommandHandler``: ``(args: str, ctx: CommandContext) -> CommandResult``.

These are extracted from the old ``cli.py`` if/elif chain.
"""

from __future__ import annotations

from datetime import datetime

from miniharness.commands.types import CommandContext, CommandResult


# ---------------------------------------------------------------------------
# Session control
# ---------------------------------------------------------------------------


def cmd_exit(args: str, ctx: CommandContext) -> CommandResult:
    """Exit the REPL."""
    return CommandResult.done("Goodbye!")


def cmd_clear(args: str, ctx: CommandContext) -> CommandResult:
    """Clear conversation history."""
    ctx.loop.clear()
    return CommandResult.refreshed("Conversation cleared.")


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------


def cmd_help(args: str, ctx: CommandContext) -> CommandResult:
    """Show all available commands."""
    from rich.table import Table

    table = Table(title="Available Commands", show_header=True)
    table.add_column("Command", style="bold")
    table.add_column("Description")

    # We need access to the registry from the context.
    # The registry is passed via a closure when registering commands.
    # For now, show a static help message.
    lines = ["**Commands**", ""]
    lines.append("  /exit, /quit, /q    Exit MiniHarness")
    lines.append("  /clear              Clear conversation history")
    lines.append("  /history            Show message count")
    lines.append("  /model [name]       Show or switch the model")
    lines.append("  /turns [n]          Show or set max turns")
    lines.append("  /permissions [mode] Show / cycle / set permission mode")
    lines.append("  /temperature [n]    Show or set LLM temperature")
    lines.append("  /top-p [n]          Show or set LLM top_p")
    lines.append("  /max-tokens [n]     Show or set max output tokens")
    lines.append("  /memory             Show core/semantic/episodic memory")
    lines.append("  /sessions           List saved sessions")
    lines.append("  /resume [id]        Resume a saved session")
    lines.append("  /tag <name>         Tag current session")
    lines.append("  /hooks              Show active hook configuration")
    lines.append("  /skills             List available skills (source + status)")
    lines.append("  /plugins [name]     List/inspect/toggle plugins (on|off)")
    lines.append("  /tools              List, describe, or execute tools")
    lines.append("  /mcp                Show MCP server connection status")
    lines.append("  /help               Show this help")
    lines.append("")
    lines.append("**Skill Commands** (type the skill name to invoke)")
    if ctx.skill_registry is not None:
        for skill in ctx.skill_registry.list_skills():
            if skill.user_invocable:
                lines.append(f"  /{skill.name:<20} {skill.description[:60]}")
    lines.append("")
    lines.append("Anything else is sent to the model as a prompt.")

    return CommandResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


def cmd_history(args: str, ctx: CommandContext) -> CommandResult:
    """Show message count."""
    count = len(ctx.loop.conversation.messages)
    return CommandResult.ok(f"Conversation has {count} messages (including system prompt).")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def cmd_model(args: str, ctx: CommandContext) -> CommandResult:
    """Show or switch the active model."""
    if not args:
        return CommandResult.ok(f"Current model: {ctx.loop.model}")
    ctx.loop.set_model(args)
    return CommandResult.refreshed(f"Model switched to {args}.")


# ---------------------------------------------------------------------------
# Turns
# ---------------------------------------------------------------------------


def cmd_turns(args: str, ctx: CommandContext) -> CommandResult:
    """Show or set max turns."""
    if not args:
        return CommandResult.ok(f"Max turns: {ctx.loop.settings.max_turns}")
    if not args.isdigit() or int(args) < 1:
        return CommandResult.ok(f"Invalid turn count '{args}' — must be a positive integer.")
    ctx.loop.settings.max_turns = int(args)
    return CommandResult.refreshed(f"Max turns set to {ctx.loop.settings.max_turns}.")


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------


def cmd_permissions(args: str, ctx: CommandContext) -> CommandResult:
    """Show, cycle, or set permission mode."""
    labels = {
        "default": "ask before write & shell",
        "accept-edits": "auto-allow writes, ask before shell",
        "bypass": "allow everything (no prompts)",
        "plan": "read-only (deny all writes & shell)",
    }
    if not args:
        mode = ctx.loop.permissions.mode
        lines = [f"Current permission mode: {mode} — {labels.get(mode, '')}", ""]
        lines.append("Available modes:")
        for m, desc in labels.items():
            lines.append(f"  /permissions {m:<12} | {desc}")
        return CommandResult.ok("\n".join(lines))

    if args not in labels:
        return CommandResult.ok(
            f"Unknown mode '{args}'. Valid: {', '.join(labels)}"
        )
    ctx.loop.permissions.mode = args
    return CommandResult.refreshed(f"Permission mode: {args} — {labels[args]}")


# ---------------------------------------------------------------------------
# LLM sampling params
# ---------------------------------------------------------------------------


def cmd_temperature(args: str, ctx: CommandContext) -> CommandResult:
    """Show or set LLM temperature."""
    if not args:
        val = ctx.loop.settings.agent.temperature
        if val is None:
            return CommandResult.ok("Temperature: unset (using provider default)")
        return CommandResult.ok(f"Temperature: {val}")
    try:
        val = float(args)
    except ValueError:
        return CommandResult.ok(f"Invalid temperature '{args}' — must be a number (0.0–2.0).")
    ctx.loop.settings.agent.temperature = val
    return CommandResult.refreshed(f"Temperature set to {val}.")


def cmd_top_p(args: str, ctx: CommandContext) -> CommandResult:
    """Show or set nucleus sampling threshold."""
    if not args:
        val = ctx.loop.settings.agent.top_p
        if val is None:
            return CommandResult.ok("Top-p: unset (using provider default)")
        return CommandResult.ok(f"Top-p: {val}")
    try:
        val = float(args)
    except ValueError:
        return CommandResult.ok(f"Invalid top-p '{args}' — must be a number (0.0–1.0).")
    ctx.loop.settings.agent.top_p = val
    return CommandResult.refreshed(f"Top-p set to {val}.")


def cmd_max_tokens(args: str, ctx: CommandContext) -> CommandResult:
    """Show or set max output tokens."""
    if not args:
        val = ctx.loop.settings.agent.max_tokens
        if val is None:
            return CommandResult.ok("Max tokens: unset (using provider default)")
        return CommandResult.ok(f"Max tokens: {val}")
    if not args.isdigit() or int(args) < 1:
        return CommandResult.ok(f"Invalid max tokens '{args}' — must be a positive integer.")
    ctx.loop.settings.agent.max_tokens = int(args)
    return CommandResult.refreshed(f"Max tokens set to {args}.")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def cmd_memory(args: str, ctx: CommandContext) -> CommandResult:
    """Display core, semantic, and episodic memory."""
    from miniharness.memory.episodic import EpisodicStore
    from miniharness.memory.semantic import SemanticStore

    cm = ctx.loop.core_memory
    lines = [f"**Core Memory**  ({cm.path})", "", cm.read(), ""]

    # Semantic.
    sem = SemanticStore(str(ctx.loop.cwd))
    entries = sem.list_all()[:5]
    lines.append("**Semantic Memory** (last 5)")
    if entries:
        for e in entries:
            tags = ", ".join(e.get("tags", []))
            tag_str = f" ({tags})" if tags else ""
            lines.append(f"  • {e['fact']}{tag_str}")
    else:
        lines.append("  (empty)")
    lines.append("")

    # Episodic.
    epi = EpisodicStore(str(ctx.loop.cwd))
    entries = epi.list_all()[:5]
    lines.append("**Episodic Memory** (last 5)")
    if entries:
        for e in entries:
            ts = datetime.fromtimestamp(e.get("timestamp", 0)).strftime("%m-%d %H:%M")
            lines.append(f"  [{ts}] {e['task']}")
            lines.append(f"  {e['summary'][:120]}")
    else:
        lines.append("  (empty)")

    return CommandResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------


def cmd_hooks(args: str, ctx: CommandContext) -> CommandResult:
    """Show active hook configuration."""
    from miniharness.config.settings import HookSettings

    hs = ctx.loop.settings.hooks
    if not isinstance(hs, HookSettings):
        hs = HookSettings()

    lines = ["**Hook Configuration**", ""]
    lines.append(f"  Dangerous commands:  {'ON' if hs.dangerous_commands else 'OFF'}")
    lines.append(f"  Sensitive files:     {'ON' if hs.sensitive_files else 'OFF'}")
    lines.append(f"  Human approval:      {'ON' if hs.human_approval else 'OFF'}")
    lines.append(f"  Audit log:           {'ON' if hs.audit_log else 'OFF'} ({hs.audit_log_dir})")
    lines.append(f"  Code security (LLM): {'ON' if hs.code_security_review else 'OFF'}")
    lines.append("")

    if ctx.hook_registry is not None:
        lines.append(f"**Active Hooks** ({ctx.hook_registry.total_count} total)")
        lines.append(ctx.hook_registry.summary() or "  (none)")

    return CommandResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Skills list
# ---------------------------------------------------------------------------


def cmd_mcp(args: str, ctx: CommandContext) -> CommandResult:
    """Show MCP server status, distinguishing direct vs plugin sources."""
    if ctx.loop is None:
        return CommandResult.ok("MCP manager not available.")

    mcp = getattr(ctx.loop, 'mcp_manager', None)
    if mcp is None:
        return CommandResult.ok("MCP manager not available.")

    statuses = mcp.list_statuses()
    if not statuses:
        return CommandResult.ok("No MCP servers configured.")

    icons = {"connected": "✅", "pending": "⏳", "failed": "❌", "disabled": "⚫"}
    lines = ["**MCP Servers**", ""]

    # Group: direct vs plugin.
    direct = [s for s in statuses if ":" not in s.name]
    plugin_servers = [s for s in statuses if ":" in s.name]

    if direct:
        lines.append("── Direct (settings / mcp.json) ──")
        for s in direct:
            lines.append(f"  {icons.get(s.state, '?')} {s.name} [{s.state}]")
            if s.tools:
                names = ", ".join(t.name for t in s.tools[:5])
                lines.append(f"     Tools: {len(s.tools)} ({names}{'...' if len(s.tools) > 5 else ''})")

    if plugin_servers:
        lines.append("")
        lines.append("── Plugin-contributed ──")
        for s in plugin_servers:
            plugin_name = s.name.split(":", 1)[0]
            active = _is_plugin_active(ctx, plugin_name)
            p_icon = "🟢" if active else "🔴"
            lines.append(f"  {icons.get(s.state, '?')} {p_icon} {s.name} [{s.state}]")
            if s.tools:
                names = ", ".join(t.name for t in s.tools[:3])
                lines.append(f"     Tools: {len(s.tools)} ({names}{'...' if len(s.tools) > 3 else ''})")

    if not direct and not plugin_servers:
        lines.append("  (none)")

    return CommandResult.ok("\n".join(lines))


def cmd_tools(args: str, ctx: CommandContext) -> CommandResult:
    """List, describe, or execute tools.

    Usage:
        /tools               — list all tools
        /tools <name>        — show a tool's input schema
        /tools <name> <json> — execute a tool with JSON arguments
    """
    if ctx.tool_registry is None:
        return CommandResult.ok("Tool registry not available.")

    tools = ctx.tool_registry.to_openai_tools()

    # No args: list all tools.
    if not args:
        lines = [f"**Available Tools** ({len(tools)} total)", ""]
        for t in tools:
            fn = t.get("function", {})
            name = fn.get("name", "?")
            desc = fn.get("description", "").split("\n")[0]
            max_desc = 80
            if len(desc) > max_desc:
                desc = desc[:max_desc - 3] + "..."
            lines.append(f"  {name:<20} {desc}")
        return CommandResult.ok("\n".join(lines))

    # Parse: "/tools bash" or '/tools bash {"command": "ls"}'
    parts = args.split(maxsplit=1)
    tool_name = parts[0].strip()
    json_args = parts[1].strip() if len(parts) > 1 else ""

    # Find the tool.
    tool = ctx.tool_registry.get(tool_name)
    if tool is None:
        similar = [t.get("function", {}).get("name", "") for t in tools
                   if tool_name.lower() in t.get("function", {}).get("name", "")]
        hint = f" Did you mean: {', '.join(similar)}?" if similar else ""
        return CommandResult.ok(f"Tool '{tool_name}' not found.{hint}")

    # Just show schema if no JSON args provided.
    if not json_args:
        fn = tool.to_openai_tool().get("function", {})
        props = fn.get("parameters", {}).get("properties", {})
        required = fn.get("parameters", {}).get("required", [])

        lines = [f"**{tool.name}**", ""]
        lines.append(f"  Description: {tool.description}")
        lines.append("  Parameters:")
        if props:
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                lines.append(f"    {pname}: {ptype}{req}")
                if pdesc:
                    lines.append(f"      {pdesc[:100]}")
        else:
            lines.append("    (none)")
        lines.append("")
        lines.append(f"  Usage: /tools {tool.name} '{{\"<param>\": \"<value>\"}}'")
        return CommandResult.ok("\n".join(lines))

    # Execute tool with JSON args.
    import asyncio
    import json
    try:
        arguments = json.loads(json_args) if json_args else {}
    except json.JSONDecodeError as exc:
        return CommandResult.ok(f"Invalid JSON: {exc}")

    if not isinstance(arguments, dict):
        return CommandResult.ok("Arguments must be a JSON object (dictionary).")

    try:
        result = asyncio.run(ctx.tool_registry.execute(tool_name, arguments))
    except Exception as exc:
        return CommandResult.ok(f"Tool execution error: {exc}")

    output = result.output
    if len(output) > 2000:
        output = output[:2000] + f"\n...(truncated, {len(result.output)} total chars)"
    status = "[error]" if result.is_error else "[ok]"
    return CommandResult.ok(f"**/{tool_name}** {status}\n\n{output}")


def cmd_plugins(args: str, ctx: CommandContext) -> CommandResult:
    """List all plugins with their contents and activation status.

    /plugins              → list all plugins
    /plugins <name>       → show details of one plugin
    /plugins <name> on    → activate plugin in this conversation
    /plugins <name> off   → deactivate plugin in this conversation
    """
    plugin_index = _get_plugin_index(ctx)
    if plugin_index is None:
        return CommandResult.ok("Plugin system not available.")

    if not plugin_index:
        return CommandResult.ok(
            "No plugins installed.\n\n"
            "Add plugins to ~/.miniharness/plugins/ or .miniharness/plugins/"
        )

    # Parse: /plugins <name> [on|off]
    parts = args.split(maxsplit=1) if args else []
    name = parts[0].strip() if parts else ""
    action = parts[1].strip().lower() if len(parts) > 1 else ""

    # ── Toggle: /plugins <name> on|off ───────────────────────────
    if name and action in ("on", "off"):
        for entry in plugin_index:
            if entry["name"] == name:
                was_active = entry.get("active", False)
                entry["active"] = (action == "on")
                if was_active == entry["active"]:
                    return CommandResult.ok(f"Plugin '{name}' is already {'active' if was_active else 'inactive'}.")
                return CommandResult.ok(f"Plugin '{name}' {'activated' if action == 'on' else 'deactivated'}.")
        return CommandResult.ok(f"Plugin '{name}' not found.")

    # ── Detail: /plugins <name> ──────────────────────────────────
    if name and not action:
        for entry in plugin_index:
            if entry["name"] != name:
                continue
            active = entry.get("active", False)
            skills = entry.get("skills", [])
            plugin_obj = entry.get("_plugin")
            hooks = getattr(plugin_obj, "hooks", {}) if plugin_obj else {}
            mcp = getattr(plugin_obj, "mcp_servers", {}) if plugin_obj else {}

            icon = "🟢" if active else "🔴"
            status = "ACTIVE" if active else "INACTIVE"
            lines = [
                f"{icon} **{name}** [{status}]",
                f"  Description: {entry.get('description', '(none)')}",
                f"  Enabled: {'yes' if getattr(plugin_obj, 'enabled', False) else 'no'}",
                "  Active: controls current conversation prompt/tool exposure",
                "",
            ]
            lines.append(f"  Skills ({len(skills)}):")
            for s in skills:
                lines.append(f"    - /{s.invocation_name}: {s.description[:60]}")
            if not skills:
                lines.append("    (none)")

            hook_count = sum(len(v) for v in hooks.values())
            lines.append(f"  Hooks: {hook_count}")
            lines.append(f"  MCP Servers: {len(mcp)}")
            for srv_name in mcp:
                lines.append(f"    - {name}:{srv_name}")

            lines.append("")
            lines.append(f"  /plugins {name} on   → activate")
            lines.append(f"  /plugins {name} off  → deactivate")
            return CommandResult.ok("\n".join(lines))

        return CommandResult.ok(f"Plugin '{name}' not found.")

    # ── List: /plugins (no args) ─────────────────────────────────
    lines = [f"**Plugins** ({len(plugin_index)} total)", ""]
    for entry in plugin_index:
        active = entry.get("active", False)
        icon = "🟢" if active else "🔴"
        skills_n = len(entry.get("skills", []))
        plugin_obj = entry.get("_plugin")
        hooks = getattr(plugin_obj, "hooks", {}) if plugin_obj else {}
        hooks_n = sum(len(v) for v in hooks.values())
        mcp_n = len(getattr(plugin_obj, "mcp_servers", {})) if plugin_obj else 0

        desc = (entry.get("description") or "")[:60]
        status = "ACTIVE" if active else "INACTIVE"
        lines.append(f"  {icon} {entry['name']:<25} [{status}]")
        if desc:
            lines.append(f"     {desc}")
        lines.append(f"     skills:{skills_n}  hooks:{hooks_n}  mcp:{mcp_n}")
    lines.append("")
    lines.append("enabled = trusted/loaded   active = visible in this conversation")
    lines.append("🟢 = active (skills/MCP visible)   🔴 = inactive (hidden and execution-blocked)")
    lines.append("/plugins <name> for details, /plugins <name> on|off to toggle")
    return CommandResult.ok("\n".join(lines))


def cmd_skills(args: str, ctx: CommandContext) -> CommandResult:
    """List all skills with source and plugin activation status."""
    if ctx.skill_registry is None:
        return CommandResult.ok("Skill registry not available.")

    skills = ctx.skill_registry.list_skills()
    if not skills:
        return CommandResult.ok("No skills available.")

    # Group direct skills by source; plugin skills get one section per plugin.
    source_order = ["bundled", "project", "user"]
    source_labels = {"bundled": "Built-in", "project": "Project", "user": "User"}
    grouped: dict[str, list] = {k: [] for k in source_order}
    plugin_grouped: dict[str, list] = {}
    for s in skills:
        if s.source == "plugin":
            plugin_grouped.setdefault(getattr(s, "plugin_name", "") or "(unknown)", []).append(s)
        else:
            grouped.setdefault(s.source, []).append(s)

    lines = [f"**Available Skills** ({len(skills)} total)", ""]
    for src in source_order:
        group = grouped.get(src, [])
        if not group:
            continue
        lines.append(f"── {source_labels.get(src, src)} ──")
        for s in group:
            lines.append(f"  /{s.invocation_name:<25} {s.description[:50]}")
        lines.append("")

    for plugin_name in sorted(plugin_grouped):
        active = _is_plugin_active(ctx, plugin_name)
        icon = "🟢" if active else "🔴"
        lines.append(f"── {plugin_name} ──")
        for s in plugin_grouped[plugin_name]:
            lines.append(f"  /{s.invocation_name:<25} {s.description[:50]} {icon}")
        lines.append("")

    lines.append("Use /<skill-name> to invoke, or /plugins to manage plugins.")
    return CommandResult.ok("\n".join(lines))


# ---------------------------------------------------------------------------
# Plugin helpers (shared by cmd_mcp, cmd_skills, cmd_plugins)
# ---------------------------------------------------------------------------


def _get_plugin_index(ctx: CommandContext) -> list[dict] | None:
    """Get the plugin index from the loop."""
    loop = getattr(ctx, "loop", None)
    if loop is None:
        return None
    return getattr(loop, "_plugin_index", None)


def _is_plugin_active(ctx: CommandContext, plugin_name: str) -> bool:
    """Check if a plugin is active in the plugin index."""
    index = _get_plugin_index(ctx)
    if not index:
        return False
    for entry in index:
        if entry.get("name") == plugin_name:
            return entry.get("active", False)
    return False
