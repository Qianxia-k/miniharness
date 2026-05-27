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

load_dotenv()

from miniharness.config import apply_cli_overrides, load_settings
from miniharness.config.settings import Settings
from miniharness.loop import AgentLoop
from miniharness.providers import get_profile


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
    sandbox: bool | None = typer.Option(None, "--sandbox/--no-sandbox", help="Enable/disable Docker sandbox"),
    sandbox_image: str | None = typer.Option(None, "--sandbox-image", help="Docker image for sandbox"),
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
        sandbox=sandbox,
        sandbox_image=sandbox_image,
    )

    if dry_run:
        _print_dry_run(root, settings, prompt or "")
        raise typer.Exit(0)

    if prompt is None:
        # No prompt argument → interactive REPL mode
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


async def _run_repl(*, root: Path, settings: Settings) -> None:
    """Interactive REPL: read prompts in a loop with persistent conversation.

    Mirrors OpenHarness's handle_line() dispatch: lines starting with "/"
    are treated as local commands; everything else is sent to the model.
    """
    loop = AgentLoop(cwd=root, settings=settings)

    console.print("[bold]MiniHarness[/bold] — interactive mode")
    console.print("Type [dim]/help[/dim] for commands, [dim]/exit[/dim] to quit.\n")

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

            if line.startswith("/"):
                if _handle_repl_command(line, loop, settings):
                    break
            else:
                await loop.run(line)
                console.print()  # blank line between turns
    finally:
        if settings.sandbox.enabled:
            from miniharness.sandbox import stop_sandbox

            await stop_sandbox()


def _handle_repl_command(line: str, loop: AgentLoop, settings: Settings) -> bool:
    """Handle a slash command.  Returns True if the REPL should exit.

    Mirrors OpenHarness's command dispatch in handle_line().
    """
    cmd, *rest = line.split(maxsplit=1)
    cmd = cmd.lower()

    if cmd in ("/exit", "/quit", "/q"):
        console.print("Goodbye!")
        return True

    if cmd == "/clear":
        loop.clear()
        console.print("[dim]Conversation cleared.[/dim]")

    elif cmd == "/help":
        console.print("Commands:")
        console.print("  [bold]/exit, /quit, /q[/bold]   Exit MiniHarness")
        console.print("  [bold]/clear[/bold]             Clear conversation history")
        console.print("  [bold]/history[/bold]           Show message count in conversation")
        console.print("  [bold]/help[/bold]              Show this help")
        console.print()
        console.print("Anything else is sent to the model as a prompt.")

    elif cmd == "/history":
        count = len(loop.conversation.messages)
        console.print(f"[dim]Conversation has {count} messages (including system prompt).[/dim]")

    else:
        console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
        console.print("[dim]Type /help for available commands.[/dim]")

    return False
