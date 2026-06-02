"""Dynamic system prompt assembly (Round 2.4 + 3.2).

Production-grade system prompt that mirrors OpenHarness's
``build_runtime_system_prompt()``:

1. **Environment info** — OS, shell, date, cwd, platform capabilities
   injected so the model can generate platform-correct commands.
2. **Core Memory** — always injected (stable project context).
3. **On-demand memory retrieval** — semantic + episodic entries that
   match the user's current query are injected, reducing token waste
   from irrelevant memories.
4. **Working set context** — brief summary of active files and task focus.

Usage::

    from miniharness.prompts import assemble_system_prompt

    prompt = assemble_system_prompt(
        base_prompt=SYSTEM_PROMPT,
        cwd=Path.cwd(),
        core_memory=core_memory,
        user_query="Implement JWT middleware",
    )
"""

from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from miniharness.memory.episodic import EpisodicStore
from miniharness.memory.semantic import SemanticStore

# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------


def get_environment_info(*, cwd: Path | None = None) -> str:
    """Collect platform / environment metadata for the system prompt.

    Mirrors OpenHarness's environment info block so the model knows what
    OS, shell, and date context it's operating in.
    """
    cwd_str = str(cwd or Path.cwd())
    home = str(Path.home())

    # Platform detection (mirrors OpenHarness platforms.py).
    system = platform.system()
    if system == "Darwin":
        os_name = "macOS"
    elif system == "Linux":
        # Detect WSL.
        if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
            os_name = "Linux (WSL)"
        else:
            os_name = "Linux"
    elif system == "Windows":
        os_name = "Windows"
    else:
        os_name = system

    shell = os.environ.get("SHELL", os.environ.get("COMSPEC", "unknown"))
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    local_date = datetime.now().strftime("%Y-%m-%d")

    lines = [
        "<env>",
        f"Working directory: {cwd_str}",
        f"Home: {home}",
        f"Platform: {system}",
        f"OS Version: {os_name} {platform.release()}",
        f"Shell: {shell}",
        f"Workspace Folder: {cwd_str}",
        f"Current date: {local_date}",
        f"Note: Prefer using absolute paths over relative paths as tool call args when possible.",
        "</env>",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# On-demand memory retrieval (Round 3.2)
# ---------------------------------------------------------------------------


def select_relevant_memories(
    *,
    user_query: str,
    cwd: str | Path,
    max_semantic: int = 5,
    max_episodic: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retrieve semantic and episodic memories relevant to *user_query*.

    Unlike the old approach (inject ALL core memory into every prompt),
    this function uses keyword matching to find only the entries that
    relate to the current task.  This saves tokens and improves focus.

    Returns ``(semantic_entries, episodic_entries)`` — each list is
    sorted by relevance (highest first), limited to the given max sizes.

    Parameters
    ----------
    user_query:
        The user's latest prompt — used to extract search keywords.
    cwd:
        Project root for per-project memory isolation.
    max_semantic:
        Max semantic memory entries to return.
    max_episodic:
        Max episodic memory entries to return.
    """
    semantic_entries: list[dict[str, Any]] = []
    episodic_entries: list[dict[str, Any]] = []

    if not user_query or not user_query.strip():
        return semantic_entries, episodic_entries

    try:
        cwd_str = str(Path(cwd).resolve())
        sem_store = SemanticStore(cwd_str)
        epi_store = EpisodicStore(cwd_str)

        semantic_entries = sem_store.search(user_query, limit=max_semantic)
        episodic_entries = epi_store.search(user_query, limit=max_episodic)
    except Exception:
        # Memory retrieval is best-effort — if it fails, we still have
        # the base system prompt + core memory.
        pass

    return semantic_entries, episodic_entries


def format_memories_for_prompt(
    *,
    semantic_entries: list[dict[str, Any]],
    episodic_entries: list[dict[str, Any]],
) -> str:
    """Render retrieved memories as a compact system-prompt block.

    Returns an empty string if no entries are available.
    """
    if not semantic_entries and not episodic_entries:
        return ""

    lines: list[str] = []

    if semantic_entries:
        lines.append("[Relevant Project Knowledge]")
        for entry in semantic_entries:
            tags = entry.get("tags", [])
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"  • {entry['fact']}{tag_str}")

    if episodic_entries:
        if lines:
            lines.append("")
        lines.append("[Relevant Past Experiences]")
        for entry in episodic_entries:
            ts = entry.get("timestamp", 0)
            try:
                date_str = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
            except (OSError, ValueError):
                date_str = "unknown"
            lines.append(f"  • [{date_str}] {entry.get('task', '')}")
            summary = entry.get("summary", "")
            if summary:
                lines.append(f"    {summary[:160]}")

    if not lines:
        return ""

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------


def assemble_system_prompt(
    *,
    base_prompt: str,
    cwd: Path,
    core_memory_text: str = "",
    user_query: str = "",
    tool_count: int = 0,
) -> str:
    """Assemble the full system prompt for a turn.

    Sections (in order):
    1. Base instructions (the static SYSTEM_PROMPT)
    2. Environment info (OS, shell, date, cwd)
    3. Core Memory (stable project context)
    4. Relevant memories (on-demand, keyword-matched to query)

    Parameters
    ----------
    base_prompt:
        The static system prompt (e.g., ``SYSTEM_PROMPT`` constant).
    cwd:
        Working directory.
    core_memory_text:
        Rendered core memory text (already cleaned by ``CoreMemory``).
    user_query:
        The user's latest prompt, used for memory relevance matching.
    tool_count:
        Number of available tools (informational, included in env info).

    Returns
    -------
    str
        The fully assembled system prompt, ready to be set as the first
        message's content.
    """
    sections: list[str] = [base_prompt]

    # 1. Environment info.
    env_info = get_environment_info(cwd=cwd)
    sections.append(env_info)

    # 2. Tool availability hint.
    if tool_count > 0:
        sections.append(
            f"You have access to {tool_count} tools. Use them to read, write, "
            f"search, and execute code in the workspace."
        )

    # 3. Core Memory (stable context, always injected).
    if core_memory_text:
        sections.append(core_memory_text)

    # 4. On-demand memory retrieval (Round 3.2).
    if user_query:
        sem, epi = select_relevant_memories(user_query=user_query, cwd=cwd)
        memory_block = format_memories_for_prompt(
            semantic_entries=sem, episodic_entries=epi
        )
        if memory_block:
            sections.append(memory_block)

    return "\n\n".join(sections)
