"""Project Instructions — MINIHARNESS.md loading.

Written by project maintainers to tell the agent *how* to work in this
project.  Defines rules, conventions, and constraints that the model
should follow.

Difference from CoreMemory (``.miniharness/memory/core.md``):

    ┌────────────────────┬───────────────────────────────┐
    │  CoreMemory         │  Project Instructions         │
    ├────────────────────┼───────────────────────────────┤
    │  Agent accumulated  │  Human written                │
    │  "we refactored     │  "use pathlib, not os.path;   │
    │   auth using JWT    │   tests go in tests/; never   │
    │   last session"     │   commit .env files"          │
    ├────────────────────┼───────────────────────────────┤
    │  User-level          │  Project-level               │
    │  ~/.miniharness/     │  <project>/MINIHARNESS.md    │
    └────────────────────┴───────────────────────────────┘

File search priority (first found wins):

    1. ``MINIHARNESS.md``        — MiniHarness standard
    2. ``CLAUDE.md``             — Claude Code compatibility
    3. ``AGENTS.md``             — generic alternative
    4. ``.miniharness/MINIHARNESS.md`` — subdirectory variant

Content is truncated to 12 000 characters to avoid blowing the context
window.
"""

from __future__ import annotations

from pathlib import Path

_MAX_CHARS = 12_000

_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("MINIHARNESS.md", "MINIHARNESS.md"),
    ("CLAUDE.md", "CLAUDE.md"),
    ("AGENTS.md", "AGENTS.md"),
    (".miniharness/MINIHARNESS.md", "MINIHARNESS.md"),
)


def load_project_instructions(cwd: str | Path) -> str | None:
    """Load project instructions from the project root.

    Searches for files in priority order, returns the content of the
    first one found.

    Returns ``None`` if no instructions file exists.
    """
    root = Path(cwd).resolve()

    for rel_path, display_name in _CANDIDATES:
        path = root / rel_path
        if not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not content.strip():
            continue
        return _render(content, display_name)

    return None


def get_instructions_path(cwd: str | Path) -> Path | None:
    """Return the filesystem path of the first-found instructions file.

    Returns ``None`` if no instructions file exists.
    """
    root = Path(cwd).resolve()
    for rel_path, _ in _CANDIDATES:
        path = root / rel_path
        if path.is_file():
            return path
    return None


def create_default(cwd: str | Path) -> Path:
    """Create a default ``MINIHARNESS.md`` with a template.

    Does NOT overwrite an existing file.  Returns the path to the file.
    """
    path = Path(cwd).resolve() / "MINIHARNESS.md"
    if path.exists():
        return path

    path.write_text(_DEFAULT_TEMPLATE, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _render(content: str, display_name: str) -> str:
    if len(content) > _MAX_CHARS:
        content = content[:_MAX_CHARS] + (
            f"\n\n...(truncated, {len(content)} total chars)"
        )

    return (
        f"# Project Instructions (from {display_name})\n\n"
        f"These are rules, conventions, and guidelines defined by the "
        f"project maintainers.  Follow them strictly.\n\n"
        f"{content.strip()}"
    )


_DEFAULT_TEMPLATE = """# Project Instructions

<!--
  This file tells MiniHarness (and compatible agents) how to work in
  this project.  Edit it freely — it's your project's constitution.

  What to include:
  - Build / test commands
  - Code style rules
  - Architecture constraints
  - File / directory conventions
  - Security rules (e.g. "never log credentials")

  See also:
  - Core Memory: ~/.miniharness/memory/core.md (agent-accumulated knowledge)
  - Skills: .miniharness/skills/ (task-specific instruction packs)
-->

## Build & Test
- Build: `uv build`
- Test: `uv run pytest tests/`
- Lint: `uv run ruff check src/`

## Code Style
- Use type hints on all public functions.
- Use `pathlib.Path` for file paths.
- Max line length: 100 characters.

## Project Structure
- Source code: `src/`
- Tests: `tests/`
- Configuration: `config/`

## Security
- Never commit secrets (.env, credentials, API keys).
- Use environment variables for sensitive configuration.
"""
