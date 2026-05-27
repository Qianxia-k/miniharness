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
    """Run a single MiniHarness prompt."""
    if prompt is None:
        console.print("[bold]MiniHarness[/bold]")
        console.print("Usage: uv run mh \"summarize this project\"")
        raise typer.Exit(0)

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
        _print_dry_run(root, settings, prompt)
        raise typer.Exit(0)

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
