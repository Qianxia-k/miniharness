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

    # Group by source.
    builtins: list[tuple[str, str]] = []
    skills: list[tuple[str, str]] = []

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
    lines.append("  /skills             List available skills")
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
    lines.append(f"**Semantic Memory** (last 5)")
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
    lines.append(f"**Episodic Memory** (last 5)")
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


def cmd_skills(args: str, ctx: CommandContext) -> CommandResult:
    """List all available skills."""
    if ctx.skill_registry is None:
        return CommandResult.ok("Skill registry not available.")

    skills = ctx.skill_registry.list_skills()
    if not skills:
        return CommandResult.ok("No skills available.")

    lines = [f"**Available Skills** ({len(skills)} total)", ""]
    for s in skills:
        source_tag = f"[{s.source}]"
        invocable = "" if s.user_invocable else " (model-only)"
        lines.append(f"  /{s.name:<25} {source_tag} {s.description[:60]}{invocable}")
    lines.append("")
    lines.append("Use /<skill-name> to invoke a skill, or type a prompt that matches a skill description.")

    return CommandResult.ok("\n".join(lines))
