"""Command line entrypoint for MiniHarness.

Settings flow (mirrors OpenHarness's runtime.py):
    1. load_settings()  →  defaults + env vars + provider auto-detect
    2. apply_cli_overrides()  →  CLI args win over everything
    3. Start sandbox if enabled
    4. Run agent loop
    5. Stop sandbox in finally
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv
import typer
from rich.console import Console

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
        profile=profile,
        model=model,
        base_url=base_url,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        sandbox=sandbox,
        sandbox_image=sandbox_image,
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
        # --resume ID → try exact ID first, then tag, then picker
        session_id = _resolve_session_id(root, resume)
        asyncio.run(_run_repl(root=root, settings=settings, resume_session_id=session_id))
    elif prompt is None:
        # No prompt argument → interactive REPL mode (fresh session)
        asyncio.run(_run_repl(root=root, settings=settings))
    else:
        # Prompt provided → single-shot mode (original behaviour)
        asyncio.run(_run(prompt=prompt, root=root, settings=settings))


def _print_dry_run(root: Path, settings: Settings, prompt: str) -> None:
    """Print resolved settings for --dry-run."""
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


# ---------------------------------------------------------------------------
# Session list rendering — single function shared by CLI and REPL
# ---------------------------------------------------------------------------


def _render_session_list(
    sessions: list[dict],
    *,
    numbered: bool = True,
    current_id: str | None = None,
    show_header: bool = True,
) -> None:
    """Render a session list to the console.  Single source of truth."""
    if not sessions:
        console.print("[dim]No saved sessions for this project.[/dim]")
        return

    from datetime import datetime

    ID_W = 14
    MSG_W = 6
    MODEL_W = 22
    DATE_W = 12
    TAG_W = 16
    # Fixed prefix: 2-space indent + 3-char number (or 3 spaces) + 1 space = 6 chars
    PREFIX_W = 6

    if show_header:
        hdr = (
            f"{'':>{PREFIX_W}}{'会话ID':<{ID_W}} {'消息':>{MSG_W}}  "
            f"{'模型':<{MODEL_W}} {'日期':<{DATE_W}} {'备注'}"
        )
        console.print(hdr, style="bold")
        console.print(" " * PREFIX_W + "─" * (len(hdr) - PREFIX_W))

    for i, s in enumerate(sessions, 1):
        ts = datetime.fromtimestamp(s.get("updated_at", s["created_at"])).strftime("%m-%d %H:%M")
        sid = s["session_id"]
        tag = s.get("tag", "") or "-"
        marker = " ←" if sid == current_id else ""

        num = f"{i:>2}." if numbered else ""
        prefix = f"  {num:<3} "  # 2 + 3 + 1 = 6 chars

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
    """``--sessions`` CLI flag: list sessions and exit."""
    sessions = list_sessions(str(root))
    console.print("[bold]Saved sessions[/bold] (newest first):\n")
    _render_session_list(sessions, show_header=True)
    if sessions:
        console.print()
        console.print("Resume: [bold]uv run mh --resume <id>[/bold]  or  [bold]uv run mh -c[/bold]")


def _resolve_session_id(root: Path, resume_arg: str) -> str:
    """Resolve a --resume argument to a concrete session ID.

    Tries in order: exact session ID → tag name → interactive picker.
    """
    data = load_session_by_id(str(root), resume_arg)
    if data is not None:
        return resume_arg
    data = load_session_by_tag(str(root), resume_arg)
    if data is not None:
        return data.get("session_id", resume_arg)

    console.print(f"[yellow]Session '{resume_arg}' not found.[/yellow]")
    return _pick_session(root)


def _pick_session(root: Path) -> str:
    """Interactive session picker.  Returns the chosen session_id or ``"latest"``."""
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


async def _run(*, prompt: str, root: Path, settings: Settings) -> None:
    """Async entry point — manages sandbox lifecycle around the agent loop."""
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


async def _run_repl(
    *,
    root: Path,
    settings: Settings,
    resume_session_id: str | None = None,
) -> None:
    """Interactive REPL: read prompts in a loop with persistent conversation.

    Every turn is automatically saved to disk so ``--continue`` or
    ``--resume`` can pick up where the user left off.

    Mirrors OpenHarness's handle_line() + session_storage integration.
    """
    import uuid

    loop = AgentLoop(cwd=root, settings=settings)
    loop.session_id = uuid.uuid4().hex[:12]

    # ---- resume saved session -------------------------------------------
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

    console.print("[bold]MiniHarness[/bold] — interactive mode")
    console.print("Type [dim]/help[/dim] for commands, [dim]/exit[/dim] to quit.")

    # Hint about saved sessions (only for fresh sessions, not resumes).
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

    # Sandbox lifecycle for the whole session.
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
                should_exit, should_save, loop = _handle_repl_command(line, loop)
                if should_exit:
                    break
            else:
                await loop.run(line)
                console.print()  # blank line between turns
                should_save = True

            # ---- auto-save after every turn ------------------------------
            if should_save:
                save_loop_snapshot(loop)
    finally:
        if settings.sandbox.enabled:
            from miniharness.sandbox import stop_sandbox

            await stop_sandbox()


def _handle_repl_command(
    line: str,
    loop: AgentLoop,
) -> tuple[bool, bool, AgentLoop]:
    """Handle a slash command.

    Returns ``(should_exit, should_save, active_loop)``.  Read-only commands
    must not touch session files, otherwise simply listing sessions can
    unexpectedly move the active ``latest`` pointer.

    Mirrors OpenHarness's command dispatch in handle_line().
    """
    cmd, *rest = line.split(maxsplit=1)
    cmd = cmd.lower()
    arg = rest[0].strip() if rest else ""

    if cmd in ("/exit", "/quit", "/q"):
        console.print("Goodbye!")
        return True, False, loop

    if cmd == "/clear":
        loop.clear()
        console.print("[dim]Conversation cleared.[/dim]")
        return False, True, loop

    elif cmd == "/help":
        console.print("Commands:")
        console.print("  [bold]/exit, /quit, /q[/bold]   Exit MiniHarness")
        console.print("  [bold]/clear[/bold]             Clear conversation history")
        console.print("  [bold]/history[/bold]           Show message count in conversation")
        console.print("  [bold]/model [name][/bold]      Show or switch the model")
        console.print("  [bold]/turns [n][/bold]         Show or set max agent turns")
        console.print("  [bold]/permissions [mode][/bold] Show / cycle / set permission mode (default / accept-edits / bypass / plan)")
        console.print("  [bold]/temperature [n][/bold]   Show or set LLM temperature")
        console.print("  [bold]/top-p [n][/bold]         Show or set LLM top_p")
        console.print("  [bold]/max-tokens [n][/bold]    Show or set max output tokens")
        console.print("  [bold]/sessions[/bold]          List saved sessions for this project")
        console.print("  [bold]/resume [id][/bold]       Resume a saved session (no arg = picker)")
        console.print("  [bold]/tag <name>[/bold]         Tag current session with a name")
        console.print("  [bold]/help[/bold]              Show this help")
        console.print()
        console.print("Anything else is sent to the model as a prompt.")

    elif cmd == "/history":
        count = len(loop.conversation.messages)
        console.print(f"[dim]Conversation has {count} messages (including system prompt).[/dim]")

    elif cmd == "/sessions":
        sessions = list_sessions(str(loop.cwd))
        console.print("[dim]Saved sessions (newest first):[/dim]")
        _render_session_list(sessions, numbered=False, current_id=loop.session_id, show_header=True)

    elif cmd == "/model":
        _repl_model(arg, loop)

    elif cmd == "/turns":
        _repl_turns(arg, loop)

    elif cmd == "/permissions":
        _repl_permissions(arg, loop)

    elif cmd == "/temperature":
        _repl_temperature(arg, loop)

    elif cmd == "/top-p":
        _repl_top_p(arg, loop)

    elif cmd == "/max-tokens":
        _repl_max_tokens(arg, loop)

    elif cmd == "/resume":
        return False, False, _repl_resume(arg, loop)

    elif cmd == "/tag":
        return False, _repl_tag(arg, loop), loop

    else:
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return False, False, loop


def _repl_resume(arg: str, loop: AgentLoop) -> AgentLoop:
    """Handle /resume [id] in REPL mode.

    Resolves the argument to a concrete session ID, then delegates to
    :func:`~miniharness.sessions.manager.switch_session` for the
    save-and-load logic.
    """
    session_id = arg if arg else _pick_session(loop.cwd)

    if session_id == "latest" and not arg:
        return loop

    # Resolve tag / partial ID → concrete session ID.
    data = load_session_by_id(str(loop.cwd), session_id)
    if data is None:
        data = load_session_by_tag(str(loop.cwd), session_id)
    if data is None:
        console.print(f"[yellow]Session '{session_id}' not found.[/yellow]")
        return loop

    target_id = data.get("session_id", session_id)
    if loop.session_id == target_id:
        console.print(f"[dim]Already on session [bold]{target_id}[/bold].[/dim]")
        return loop

    next_loop = switch_session(loop, target_id)
    # switch_session never returns None here — we validated target_id above.
    msg_count = len(next_loop.conversation.messages)
    tag_info = f" @{next_loop.tag}" if next_loop.tag else ""
    console.print(
        f"[dim]Switched to session [bold]{next_loop.session_id}[/bold]{tag_info} "
        f"({msg_count} messages)[/dim]"
    )
    return next_loop


def _repl_tag(arg: str, loop: AgentLoop) -> bool:
    """Handle /tag <name> in REPL mode.

    Tags the current session so it can be resumed by name.
    """
    if not arg:
        console.print("[yellow]Usage: /tag <name>[/yellow]")
        return False

    if not loop.session_id:
        console.print("[yellow]No active session to tag.[/yellow]")
        return False

    ok = rename_session(str(loop.cwd), loop.session_id, arg)
    if ok:
        loop.tag = arg
        console.print(f"[dim]Session [bold]{loop.session_id}[/bold] tagged as [bold]{arg}[/bold][/dim]")
        console.print(f"[dim]Resume with: uv run mh --resume {arg}[/dim]")
        return True
    else:
        console.print(f"[yellow]Could not tag session '{loop.session_id}'.[/yellow]")
        return False


# ---------------------------------------------------------------------------
# Runtime setting commands — /model, /turns, /permissions
# ---------------------------------------------------------------------------


def _repl_model(arg: str, loop: AgentLoop) -> None:
    """Handle /model [name] — show or switch the active model.

    Without an argument, prints the current model.  With one, switches to
    the given model name immediately.
    """
    if not arg:
        console.print(f"[dim]Current model: [bold]{loop.model}[/bold][/dim]")
        return

    loop.set_model(arg)
    console.print(f"[dim]Model switched to [bold]{arg}[/bold][/dim]")


def _repl_turns(arg: str, loop: AgentLoop) -> None:
    """Handle /turns [n] — show or set the maximum agent-loop turns.

    Without an argument, prints the current value.
    """
    if not arg:
        console.print(f"[dim]Max turns: [bold]{loop.settings.max_turns}[/bold][/dim]")
        return

    if not arg.isdigit() or int(arg) < 1:
        console.print(f"[yellow]Invalid turn count '{arg}' — must be a positive integer.[/yellow]")
        return

    loop.settings.max_turns = int(arg)
    console.print(f"[dim]Max turns set to [bold]{loop.settings.max_turns}[/bold][/dim]")


def _repl_permissions(arg: str, loop: AgentLoop) -> None:
    """Handle /permissions [mode] — show or set the permission mode.

    Without an argument, cycles to the next mode.  With one, jumps directly
    to the named mode (default / accept-edits / bypass / plan).
    """
    from miniharness.permissions import PermissionMode

    labels: dict[str, str] = {
        "default": "ask before write & shell",
        "accept-edits": "auto-allow writes, ask before shell",
        "bypass": "allow everything (no prompts)",
        "plan": "read-only (deny all writes & shell)",
    }

    if not arg:
        # Show current mode and available options
        new_mode = loop.permissions.mode
        console.print(f"\n[bold]Current permission mode:[/bold] [green]{new_mode}[/green] — {labels[new_mode]}")
        console.print("\n[dim]Available modes:[/dim]")
        for mode, desc in labels.items():
            console.print(f"  [bold]/permissions {mode:<12}[/bold] | {desc}")
        console.print()
        return
    else:
        if arg not in labels:
            valid = ", ".join(sorted(labels))
            console.print(f"[yellow]Unknown mode '{arg}'. Valid modes: {valid}[/yellow]")
            return
        loop.permissions.mode = arg
        new_mode = arg

    console.print(
        f"[dim]Permission mode: [bold]{new_mode}[/bold] — {labels.get(new_mode, '')}[/dim]"
    )


# ---------------------------------------------------------------------------
# Runtime LLM param commands — /temperature, /top-p, /max-tokens
# ---------------------------------------------------------------------------


def _repl_temperature(arg: str, loop: AgentLoop) -> None:
    """Handle /temperature [value] — show or set LLM temperature."""
    if not arg:
        val = loop.settings.agent.temperature
        if val is None:
            console.print("[dim]Temperature: [bold]unset[/bold] (using provider default)[/dim]")
        else:
            console.print(f"[dim]Temperature: [bold]{val}[/bold][/dim]")
        return

    try:
        val = float(arg)
    except ValueError:
        console.print(f"[yellow]Invalid temperature '{arg}' — must be a number (0.0–2.0).[/yellow]")
        return

    loop.settings.agent.temperature = val
    console.print(f"[dim]Temperature set to [bold]{val}[/bold][/dim]")


def _repl_top_p(arg: str, loop: AgentLoop) -> None:
    """Handle /top-p [value] — show or set nucleus sampling threshold."""
    if not arg:
        val = loop.settings.agent.top_p
        if val is None:
            console.print("[dim]Top-p: [bold]unset[/bold] (using provider default)[/dim]")
        else:
            console.print(f"[dim]Top-p: [bold]{val}[/bold][/dim]")
        return

    try:
        val = float(arg)
    except ValueError:
        console.print(f"[yellow]Invalid top-p '{arg}' — must be a number (0.0–1.0).[/yellow]")
        return

    loop.settings.agent.top_p = val
    console.print(f"[dim]Top-p set to [bold]{val}[/bold][/dim]")


def _repl_max_tokens(arg: str, loop: AgentLoop) -> None:
    """Handle /max-tokens [value] — show or set max output tokens."""
    if not arg:
        val = loop.settings.agent.max_tokens
        if val is None:
            console.print("[dim]Max tokens: [bold]unset[/bold] (using provider default)[/dim]")
        else:
            console.print(f"[dim]Max tokens: [bold]{val}[/bold][/dim]")
        return

    if not arg.isdigit() or int(arg) < 1:
        console.print(f"[yellow]Invalid max tokens '{arg}' — must be a positive integer.[/yellow]")
        return

    loop.settings.agent.max_tokens = int(arg)
    console.print(f"[dim]Max tokens set to [bold]{arg}[/bold][/dim]")
