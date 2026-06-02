"""Tool output offloading — prevent large tool results from flooding the context.

When a tool returns a very large output (e.g. ``grep`` finds 5000 matches),
putting it all into the conversation would waste tokens and push the
context toward unnecessary compaction.

Instead, when output exceeds the inline threshold, we:

1. Write the **full output** to an artifact file on disk.
2. Return a **truncated inline preview** to the model with a reference to
   the artifact file.
3. The model can use ``read_file`` to read the full artifact if needed.

Mirrors OpenHarness's ``_offload_tool_output_if_needed()``.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path

# Default thresholds.
DEFAULT_INLINE_CHAR_LIMIT = 12_000   # trigger offload above this
DEFAULT_PREVIEW_CHARS = 2_000        # how much to keep inline


def tool_output_inline_chars() -> int:
    """Maximum characters before offloading kicks in."""
    return DEFAULT_INLINE_CHAR_LIMIT


def tool_output_preview_chars() -> int:
    """Characters to include as inline preview after offloading."""
    return DEFAULT_PREVIEW_CHARS


def offload_if_needed(
    *,
    tool_name: str,
    output: str,
    inline_limit: int | None = None,
    preview_chars: int | None = None,
) -> tuple[str, Path | None]:
    """Conditionally offload large tool output to disk.

    Parameters
    ----------
    tool_name:
        Name of the tool that produced the output.
    output:
        The full tool output string.
    inline_limit:
        Max chars before offloading (default: 12000).
    preview_chars:
        How many chars to keep inline as preview (default: 2000).

    Returns
    -------
    (inline_text, artifact_path | None)
        If output fits within *inline_limit*, returns ``(output, None)``.
        Otherwise returns ``(preview_text, artifact_path)`` where
        *artifact_path* points to the full output on disk.
    """
    limit = inline_limit if inline_limit is not None else DEFAULT_INLINE_CHAR_LIMIT
    preview_len = preview_chars if preview_chars is not None else DEFAULT_PREVIEW_CHARS

    if len(output) <= limit:
        return output, None

    # Write full output to artifact file.
    artifact_path = _make_artifact_path(tool_name)
    artifact_path.write_text(output, encoding="utf-8", errors="replace")

    # Build inline preview.
    preview = output[:preview_len]
    omitted = max(0, len(output) - preview_len)

    inline = (
        f"[Tool output truncated]\n"
        f"Tool: {tool_name}\n"
        f"Original size: {len(output):,} chars\n"
        f"Full output saved to: {artifact_path}\n"
        f"Inline preview (first {preview_len:,} chars"
    )
    if omitted:
        inline += f", {omitted:,} chars omitted"
    inline += "):\n\n"
    if preview:
        inline += preview
        if omitted:
            inline += f"\n\n...[{omitted:,} more chars in {artifact_path}]"

    return inline, artifact_path


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _artifact_root() -> Path:
    """Root directory for tool artifact files."""
    root = Path.home() / ".miniharness" / "tool_artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_name(tool_name: str) -> str:
    """Sanitize tool name for use in a filename."""
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", tool_name.strip())
    return normalized[:80] or "tool"


def _make_artifact_path(tool_name: str) -> Path:
    """Generate a unique, timestamped artifact file path."""
    ts = time.strftime("%Y%m%d-%H%M%S")
    uid = uuid.uuid4().hex[:12]
    filename = f"{ts}-{_safe_name(tool_name)}-{uid}.txt"
    return _artifact_root() / filename
