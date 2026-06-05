"""Command line entrypoint for MiniHarness.

Settings flow (mirrors OpenHarness's runtime.py):
    1. load_settings()  →  defaults + env vars + provider auto-detect
    2. apply_cli_overrides()  →  CLI args win over everything
    3. Build command registry (built-in + skill auto-commands)
    4. Start sandbox if enabled
    5. Run agent loop (REPL dispatch via CommandRegistry)
    6. Stop sandbox in finally
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv
import typer
from rich.console import Console

from miniharness.commands import CommandContext, CommandRegistry
from miniharness.commands.builtin import (
    cmd_clear, cmd_exit, cmd_help, cmd_history, cmd_hooks,
    cmd_max_tokens, cmd_mcp, cmd_memory, cmd_model, cmd_permissions,
    cmd_skills, cmd_temperature, cmd_tools, cmd_top_p, cmd_turns,
)
from miniharness.commands.types import CommandResult
from miniharness.config import apply_cli_overrides, load_settings
from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.providers import get_profile
from miniharness.sessions import (
    list_sessions,
    load_session_by_id,
    load_session_by_tag,
    rename_session,
    save_loop_snapshot,
    switch_session,
)


load_dotenv()

app = typer.Typer(add_completion=False, help="MiniHarness: a tiny coding agent harness.")
console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════


@app.callback(invoke_without_command=True)
def main(
    prompt: str | None = typer.Argument(None, help="Prompt to send to the agent"),
    cwd: str = typer.Option(str(Path.cwd()), "--cwd", help="Working directory for tools"),
    profile: str | None = typer.Option(None, "--profile", help="Provider profile (auto-detected if omitted)"),
    model: str | None = typer.Option(None, "--model", "-m", help="Override auto-detected model"),
    base_url: str | None = typer.Option(None, "--base-url", help="Override auto-detected base URL"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show resolved settings and exit"),
    max_turns: int | None = typer.Option(None, "--max-turns", help="Maximum agent loop turns"),
    temperature: float | None = typer.Option(None, "--temperature", help="LLM sampling temperature (0.0–2.0)"),
    top_p: float | None = typer.Option(None, "--top-p", help="LLM nucleus sampling threshold (0.0–1.0)"),
    max_tokens: int | None = typer.Option(None, "--max-tokens", help="Maximum output tokens"),
    sandbox: bool | None = typer.Option(None, "--sandbox/--no-sandbox", help="Enable/disable Docker sandbox"),
    sandbox_image: str | None = typer.Option(None, "--sandbox-image", help="Docker image for sandbox"),
    continue_session: bool = typer.Option(False, "--continue", "-c", help="Resume the most recent session"),
    resume: str | None = typer.Option(None, "--resume", help="Resume a session by ID or tag name"),
    list_sessions_flag: bool = typer.Option(False, "--sessions", help="List saved sessions and exit"),
) -> None:
    """Run a MiniHarness prompt, or start interactive REPL if no prompt given."""
    root = Path(cwd).expanduser().resolve()

    # ---- build settings -------------------------------------------------
    settings = load_settings()
    settings = apply_cli_overrides(
        settings,
        profile=profile, model=model, base_url=base_url,
        max_turns=max_turns, temperature=temperature, top_p=top_p,
        max_tokens=max_tokens, sandbox=sandbox, sandbox_image=sandbox_image,
    )

    if dry_run:
        _print_dry_run(root, settings, prompt or "")
        raise typer.Exit(0)

    if list_sessions_flag:
        _print_sessions(root)
        raise typer.Exit(0)

    if continue_session:
        asyncio.run(_run_repl(root=root, settings=settings, resume_session_id="latest"))
    elif resume is not None:
        session_id = _resolve_session_id(root, resume)
        asyncio.run(_run_repl(root=root, settings=settings, resume_session_id=session_id))
    elif prompt is None:
        asyncio.run(_run_repl(root=root, settings=settings))
    else:
        asyncio.run(_run(prompt=prompt, root=root, settings=settings))


# ═══════════════════════════════════════════════════════════════════════════
# Dry-run display
# ═══════════════════════════════════════════════════════════════════════════


def _print_dry_run(root: Path, settings: Settings, prompt: str) -> None:
    provider_profile = get_profile(settings.provider.name)
    console.print("MiniHarness dry run")
    console.print(f"- cwd: {root}")
    console.print(f"- provider: {provider_profile.name} ({provider_profile.label})")
    console.print(f"- model: {settings.provider.model or provider_profile.default_model}")
    console.print(f"- base_url: {settings.provider.base_url or provider_profile.base_url or '(provider default)'}")
    console.print(f"- max_turns: {settings.max_turns}")
    console.print(f"- sandbox.enabled: {settings.sandbox.enabled}")
    if settings.sandbox.enabled:
        console.print(f"- sandbox.image: {settings.sandbox.image}")
    console.print(f"- prompt: {prompt}")


# ═══════════════════════════════════════════════════════════════════════════
# Session list rendering — single function shared by CLI and REPL
# ═══════════════════════════════════════════════════════════════════════════


def _render_session_list(
    sessions: list[dict],
    *,
    numbered: bool = True,
    current_id: str | None = None,
    show_header: bool = True,
) -> None:
    if not sessions:
        console.print("[dim]No saved sessions for this project.[/dim]")
        return

    from datetime import datetime

    ID_W, MSG_W, MODEL_W, DATE_W, TAG_W = 14, 6, 22, 12, 16
    PREFIX_W = 6

    if show_header:
        hdr = (
            f"{'':>{PREFIX_W}}{'ID':<{ID_W}} {'Msgs':>{MSG_W}}  "
            f"{'Model':<{MODEL_W}} {'Date':<{DATE_W}} {'Tag'}"
        )
        console.print(hdr, style="bold")
        console.print(" " * PREFIX_W + "─" * (len(hdr) - PREFIX_W))

    for i, s in enumerate(sessions, 1):
        ts = datetime.fromtimestamp(s.get("updated_at", s["created_at"])).strftime("%m-%d %H:%M")
        sid = s["session_id"]
        tag = s.get("tag", "") or "-"
        marker = " ←" if sid == current_id else ""

        num = f"{i:>2}." if numbered else ""
        prefix = f"  {num:<3} "

        sid_cell = f"[bold]{sid + marker:<{ID_W}}[/bold]"
        msg_cell = f"{s['message_count']:>{MSG_W}}"
        model_cell = f"[dim]{s['model']:<{MODEL_W}}[/dim]"
        date_cell = f"[dim]{ts:<{DATE_W}}[/dim]"
        tag_cell = f"[cyan]{tag:<{TAG_W}}[/cyan]"

        console.print(f"{prefix}{sid_cell} {msg_cell}  {model_cell} {date_cell} {tag_cell}")

        summary = s["summary"][:80] if s["summary"] else ""
        if summary:
            console.print(f"{'':>{PREFIX_W}}[dim]{summary}[/dim]")


def _print_sessions(root: Path) -> None:
    sessions = list_sessions(str(root))
    console.print("[bold]Saved sessions[/bold] (newest first):\n")
    _render_session_list(sessions, show_header=True)
    if sessions:
        console.print()
        console.print("Resume: [bold]uv run mh --resume <id>[/bold]  or  [bold]uv run mh -c[/bold]")


def _resolve_session_id(root: Path, resume_arg: str) -> str:
    data = load_session_by_id(str(root), resume_arg)
    if data is not None:
        return resume_arg
    data = load_session_by_tag(str(root), resume_arg)
    if data is not None:
        return data.get("session_id", resume_arg)
    console.print(f"[yellow]Session '{resume_arg}' not found.[/yellow]")
    return _pick_session(root)


def _pick_session(root: Path) -> str:
    sessions = list_sessions(str(root))
    if not sessions:
        console.print("[dim]No saved sessions. Starting fresh.[/dim]")
        return "latest"

    console.print("[bold]Pick a session to resume:[/bold]\n")
    _render_session_list(sessions)

    try:
        choice = console.input("\nChoice (number or ID, Enter = fresh): ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return "latest"

    if not choice:
        return "latest"
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            return sessions[idx]["session_id"]

    data = load_session_by_id(str(root), choice) or load_session_by_tag(str(root), choice)
    if data is not None:
        return data.get("session_id", choice)

    console.print(f"[yellow]Invalid choice '{choice}', starting fresh.[/yellow]")
    return "latest"


# ═══════════════════════════════════════════════════════════════════════════
# Single-shot run
# ═══════════════════════════════════════════════════════════════════════════


async def _run(*, prompt: str, root: Path, settings: Settings) -> None:
    if settings.sandbox.enabled:
        from miniharness.sandbox import start_sandbox
        console.print(f"[dim]Starting sandbox (image={settings.sandbox.image})...[/dim]")
        await start_sandbox(cwd=root, image=settings.sandbox.image)
        console.print("[dim]Sandbox ready[/dim]")

    try:
        loop = AgentLoop(cwd=root, settings=settings)
        await loop.run(prompt)
    finally:
        if settings.sandbox.enabled:
            from miniharness.sandbox import stop_sandbox
            await stop_sandbox()


# ═══════════════════════════════════════════════════════════════════════════
# REPL — command registry replaces the old if/elif chain
# ═══════════════════════════════════════════════════════════════════════════


async def _run_repl(
    *,
    root: Path,
    settings: Settings,
    resume_session_id: str | None = None,
) -> None:
    loop = AgentLoop(cwd=root, settings=settings)
    loop.session_id = uuid.uuid4().hex[:12]

    # ---- Resume saved session -------------------------------------------
    if resume_session_id is not None:
        data = load_session_by_id(str(root), resume_session_id)
        if data is None:
            console.print(f"[yellow]Session '{resume_session_id}' not found, starting fresh.[/yellow]")
        else:
            loop.restore_messages(data.get("messages", []))
            loop.session_id = data.get("session_id", loop.session_id)
            loop.tag = data.get("tag", "")
            msg_count = data.get("message_count", 0)
            tag_info = f" @{loop.tag}" if loop.tag else ""
            console.print(
                f"[dim]Restored session [bold]{loop.session_id}[/bold]{tag_info} "
                f"({msg_count} messages, model={data.get('model', '?')})[/dim]"
            )

    # ---- Build command registry -----------------------------------------
    cmd_registry = _build_command_registry(loop)

    console.print("[bold]MiniHarness[/bold] — interactive mode")
    console.print("Type [dim]/help[/dim] for commands, [dim]/exit[/dim] to quit.")

    if resume_session_id is None:
        existing = list_sessions(str(root))
        if existing:
            latest = existing[0]
            console.print(
                f"[dim]{len(existing)} saved session(s). "
                f"Latest: [bold]{latest['session_id']}[/bold] "
                f"({latest['summary'][:50] if latest['summary'] else 'empty'})[/dim]"
            )
            console.print("[dim]/sessions to list, /resume to switch[/dim]")

    console.print()

    # ---- Sandbox lifecycle -----------------------------------------------
    if settings.sandbox.enabled:
        from miniharness.sandbox import start_sandbox
        console.print(f"[dim]Starting sandbox (image={settings.sandbox.image})...[/dim]")
        await start_sandbox(cwd=root, image=settings.sandbox.image)
        console.print("[dim]Sandbox ready[/dim]\n")

    try:
        while True:
            try:
                line = console.input("[bold]▸[/bold] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                break

            if not line:
                continue

            should_save = False
            if line.startswith("/"):
                # ── Dispatch via CommandRegistry ──────────────────────
                ctx = _make_command_context(loop)
                result = cmd_registry.dispatch(line, ctx)

                if result.message:
                    console.print(result.message)

                if result.exit:
                    break

                if result.submit_prompt:
                    if not await _run_repl_turn(loop, result.submit_prompt):
                        continue
                    console.print()
                    should_save = True

                # Handle /resume — it may return a new loop via ctx.
                new_loop = getattr(ctx, "_new_loop", None)
                if new_loop is not None:
                    loop = new_loop
                    cmd_registry = _build_command_registry(loop)
                    should_save = False

                if result.should_save and not result.exit:
                    should_save = True
            else:
                if not await _run_repl_turn(loop, line):
                    continue
                console.print()
                should_save = True

            if should_save:
                save_loop_snapshot(loop)
    finally:
        # Close MCP connections before event loop tears down.
        mcp = getattr(loop, '_mcp_manager', None)
        if mcp is not None:
            try:
                await mcp.close()
            except Exception:
                pass

        if settings.sandbox.enabled:
            from miniharness.sandbox import stop_sandbox
            await stop_sandbox()


async def _run_repl_turn(loop: AgentLoop, prompt: str) -> bool:
    """Run one user turn in the REPL.

    Ctrl-C during a running model/tool turn cancels that turn and returns to
    the prompt. Slash commands such as /q are only read between turns because
    the REPL owns stdin synchronously.
    """
    try:
        await loop.run(prompt)
        return True
    except asyncio.CancelledError:
        console.print("\n[yellow]Cancelled current turn.[/yellow]")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Command registry factory
# ═══════════════════════════════════════════════════════════════════════════


def _build_command_registry(loop: AgentLoop) -> CommandRegistry:
    """Build the full command registry: built-in + skill auto-commands."""
    reg = CommandRegistry()

    # ── Built-in commands ──────────────────────────────────────────────
    reg.register("exit", cmd_exit, description="Exit MiniHarness",
                 aliases=["quit", "q"], source="builtin")
    reg.register("clear", cmd_clear, description="Clear conversation history",
                 source="builtin")
    reg.register("help", cmd_help, description="Show available commands",
                 source="builtin")
    reg.register("history", cmd_history, description="Show message count",
                 source="builtin")
    reg.register("model", cmd_model, description="Show or switch the model",
                 source="builtin")
    reg.register("turns", cmd_turns, description="Show or set max turns",
                 source="builtin")
    reg.register("permissions", cmd_permissions,
                 description="Show or set permission mode", source="builtin")
    reg.register("temperature", cmd_temperature,
                 description="Show or set LLM temperature", source="builtin")
    reg.register("top-p", cmd_top_p,
                 description="Show or set LLM top_p", source="builtin")
    reg.register("max-tokens", cmd_max_tokens,
                 description="Show or set max output tokens", source="builtin")
    reg.register("memory", cmd_memory,
                 description="Show core/semantic/episodic memory", source="builtin")
    reg.register("hooks", cmd_hooks,
                 description="Show hook configuration", source="builtin")
    reg.register("skills", cmd_skills,
                 description="List available skills", source="builtin")
    reg.register("tools", cmd_tools,
                 description="List, describe, or execute tools", source="builtin")
    reg.register("mcp", cmd_mcp,
                 description="Show MCP server connection status", source="builtin")

    # ── Session commands (need closure over loop) ──────────────────────
    reg.register("sessions", _make_sessions_handler(loop.cwd),
                 description="List saved sessions", source="builtin")
    reg.register("resume", _make_resume_handler(loop),
                 description="Resume a saved session", source="builtin")
    reg.register("tag", _make_tag_handler(loop),
                 description="Tag current session", source="builtin")

    # ── Auto-generate skill slash commands ─────────────────────────────
    if hasattr(loop, 'skill_registry') and loop.skill_registry is not None:
        reg.register_from_skills(loop.skill_registry)

    return reg


def _make_command_context(loop: AgentLoop) -> CommandContext:
    return CommandContext(
        loop=loop,
        console=console,
        cwd=loop.cwd,
        skill_registry=getattr(loop, 'skill_registry', None),
        hook_registry=getattr(loop, 'hook_registry', None),
        tool_registry=getattr(loop, 'tools', None),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Session-command helpers (closures over loop for resume/tag/sessions)
# ═══════════════════════════════════════════════════════════════════════════


def _make_sessions_handler(cwd):
    def handler(args: str, ctx: CommandContext) -> CommandResult:
        from miniharness.commands.types import CommandResult
        sessions = list_sessions(str(cwd))
        _render_session_list(sessions, numbered=False,
                            current_id=ctx.loop.session_id, show_header=True)
        return CommandResult.ok()
    return handler


def _make_resume_handler(loop):
    def handler(args: str, ctx: CommandContext) -> CommandResult:
        from miniharness.commands.types import CommandResult
        session_id = args if args else _pick_session(loop.cwd)
        if session_id == "latest" and not args:
            return CommandResult.ok()

        data = load_session_by_id(str(loop.cwd), session_id)
        if data is None:
            data = load_session_by_tag(str(loop.cwd), session_id)
        if data is None:
            return CommandResult.ok(f"Session '{session_id}' not found.")

        target_id = data.get("session_id", session_id)
        if loop.session_id == target_id:
            return CommandResult.ok(f"Already on session {target_id}.")

        new_loop = switch_session(loop, target_id)
        msg_count = len(new_loop.conversation.messages)
        tag_info = f" @{new_loop.tag}" if new_loop.tag else ""
        ctx._new_loop = new_loop
        return CommandResult.ok(
            f"Switched to session {new_loop.session_id}{tag_info} ({msg_count} messages)"
        )
    return handler


def _make_tag_handler(loop):
    def handler(args: str, ctx: CommandContext) -> CommandResult:
        from miniharness.commands.types import CommandResult
        if not args:
            return CommandResult.ok("Usage: /tag <name>")
        if not loop.session_id:
            return CommandResult.ok("No active session to tag.")

        ok = rename_session(str(loop.cwd), loop.session_id, args)
        if ok:
            loop.tag = args
            return CommandResult.ok(
                f"Session {loop.session_id} tagged as '{args}'.\n"
                f"Resume with: uv run mh --resume {args}"
            )
        return CommandResult.ok(f"Could not tag session '{loop.session_id}'.")
    return handler
