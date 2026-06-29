"""YAML frontmatter parser for SKILL.md files.

A SKILL.md file looks like::

    ---
    name: my-skill
    description: Does something useful.
    model-invocable: true
    ---

    # my-skill

    Actual markdown content here...

Frontmatter is optional.  If absent, the name and description are extracted
from the first ``# Heading`` and the best descriptive prose in the body.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_skill_frontmatter(
    content: str,
    *,
    default_name: str = "unnamed",
    fallback_template: str = "Skill: {name}",
) -> dict[str, Any]:
    """Parse a SKILL.md file into its metadata components.

    Returns a dict with:
        - ``name``: canonical skill name
        - ``description``: short description
        - ``body``: markdown content without the YAML header
        - ``frontmatter``: raw frontmatter dict (empty if none)
    """
    fm: dict[str, Any] = {}
    body = content

    # 1. Try to extract YAML frontmatter (between standalone --- lines).
    parsed_fm, parsed_body = _split_frontmatter(content)
    if parsed_fm is not None:
        fm = _parse_yaml_safe(parsed_fm)
        body = parsed_body

    # 2. Extract name.
    name = _extract_str(fm.get("name"))
    if not name:
        name = _extract_name_from_body(body)
    if not name:
        name = default_name

    # 3. Extract description.
    description = _extract_str(fm.get("description"))
    if not description:
        description = _extract_description_from_body(body)
    if not description:
        description = fallback_template.format(name=name)

    return {
        "name": name.strip(),
        "description": description.strip()[:1000],
        "body": body.strip(),
        "frontmatter": fm,
    }


def parse_bool(value: Any, *, default: bool = False) -> bool:
    """Permissive boolean parser for frontmatter values.

    Accepts: True/False, "true"/"false", "yes"/"no", "1"/"0", "on"/"off".
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    return default


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_frontmatter(content: str) -> tuple[str, str] | tuple[None, str]:
    """Split YAML frontmatter from markdown body.

    The closing delimiter must be a standalone ``---`` line.  This avoids a
    common bug where ``str.split("\\n---\\n")`` terminates frontmatter at a
    later markdown horizontal rule and breaks YAML block scalars like
    ``description: >-``.
    """
    stripped = content.lstrip("\n")
    lines = stripped.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return None, content

    for idx, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[1:idx]), "".join(lines[idx + 1 :])

    return None, content


def _parse_yaml_safe(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter.  Falls back to manual key:value parsing."""
    try:
        import yaml
        result = yaml.safe_load(text)
        return _normalize_yaml_keys(result) if isinstance(result, dict) else {}
    except Exception:
        pass

    # Manual fallback: parse simple "key: value" lines plus YAML block
    # scalars (description: >-, description: |).  PyYAML is optional in
    # MiniHarness, but SKILL.md files commonly use block descriptions.
    result: dict[str, Any] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace("-", "_")
            val = val.strip().strip('"').strip("'")
            if key and val in {">", ">-", ">+", "|", "|-", "|+"}:
                block_lines: list[str] = []
                i += 1
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith((" ", "\t")):
                        i -= 1
                        break
                    block_lines.append(next_line)
                    i += 1
                result[key] = _fold_yaml_block(block_lines, literal=val.startswith("|"))
            elif key and val:
                result[key] = val
        i += 1
    return result


def _normalize_yaml_keys(value: Any) -> Any:
    """Normalize YAML mapping keys while preserving parsed YAML value types."""
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = str(key).strip().replace("-", "_")
            normalized[normalized_key] = _normalize_yaml_keys(item)
        return normalized
    if isinstance(value, list):
        return [_normalize_yaml_keys(item) for item in value]
    return value


def _fold_yaml_block(lines: list[str], *, literal: bool) -> str:
    """Fold a minimal YAML block scalar into text for frontmatter fallback."""
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return ""
    min_indent = min(len(line) - len(line.lstrip(" \t")) for line in non_empty)
    stripped_lines = [line[min_indent:].rstrip() if len(line) >= min_indent else line.rstrip() for line in lines]
    text = "\n".join(stripped_lines).strip()
    if literal:
        return text
    paragraphs = re.split(r"\n\s*\n", text)
    folded: list[str] = []
    for paragraph in paragraphs:
        folded.append(re.sub(r"\s*\n\s*", " ", paragraph).strip())
    return "\n\n".join(part for part in folded if part)


def _extract_str(value: Any) -> str:
    """Return a trimmed string, or empty string."""
    if isinstance(value, str):
        return value.strip()
    return ""


def _extract_name_from_body(body: str) -> str:
    """Extract skill name from the first ``# Heading`` in the body."""
    for line in body.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return ""


def _extract_description_from_body(body: str) -> str:
    """Extract a concise, model-useful description from markdown.

    Production skill files often start with badges, TOCs, screenshots, or a
    terse heading before the actual purpose statement.  Prefer explicit
    description/overview sections, otherwise choose the first meaningful prose
    paragraph and normalize it into one sentence-sized string.
    """
    section = _extract_named_description_section(body)
    if section:
        return section

    candidates = _extract_paragraph_candidates(body)
    if not candidates:
        return ""

    candidates.sort(key=_description_candidate_score, reverse=True)
    return _compact_description(candidates[0])


def _extract_named_description_section(body: str) -> str:
    """Return text from a likely description/overview/purpose section."""
    target_headings = {
        "description",
        "overview",
        "summary",
        "purpose",
        "when to use",
        "when to use this skill",
        "use when",
    }
    lines = body.splitlines()
    capture: list[str] = []
    in_section = False

    for line in lines:
        stripped = line.strip()
        heading = _markdown_heading_text(stripped)
        if heading is not None:
            normalized = heading.lower().strip(":")
            if in_section:
                break
            in_section = normalized in target_headings
            continue
        if in_section:
            capture.append(line)

    return _best_description_from_lines(capture)


def _extract_paragraph_candidates(body: str) -> list[str]:
    """Collect meaningful prose/list candidates from markdown body."""
    candidates: list[str] = []
    current: list[str] = []
    in_code = False

    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code = not in_code
            _flush_candidate(current, candidates)
            continue
        if in_code:
            continue
        if _should_skip_markdown_line(stripped):
            _flush_candidate(current, candidates)
            continue
        normalized = _normalize_markdown_line(stripped)
        if not normalized:
            _flush_candidate(current, candidates)
            continue
        current.append(normalized)

    _flush_candidate(current, candidates)
    return candidates


def _best_description_from_lines(lines: list[str]) -> str:
    candidates = _extract_paragraph_candidates("\n".join(lines))
    if not candidates:
        return ""
    candidates.sort(key=_description_candidate_score, reverse=True)
    return _compact_description(candidates[0])


def _flush_candidate(current: list[str], candidates: list[str]) -> None:
    if not current:
        return
    text = _compact_description(" ".join(current))
    current.clear()
    if _is_meaningful_description(text):
        candidates.append(text)


def _markdown_heading_text(stripped: str) -> str | None:
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", stripped)
    if not match:
        return None
    return re.sub(r"\s+#*$", "", match.group(1)).strip()


def _should_skip_markdown_line(stripped: str) -> bool:
    if not stripped or stripped == "---":
        return True
    if _markdown_heading_text(stripped) is not None:
        return True
    if stripped.startswith(("!", "[!", "<img", "<picture", "<!--")):
        return True
    if stripped.startswith(("```", "~~~", "|")):
        return True
    lowered = stripped.lower()
    if lowered in {"table of contents", "toc"}:
        return True
    if lowered.startswith(("- [", "* [")):
        return True
    return False


def _normalize_markdown_line(stripped: str) -> str:
    stripped = re.sub(r"^>\s*", "", stripped)
    stripped = re.sub(r"^[-*+]\s+", "", stripped)
    stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
    stripped = re.sub(r"`([^`]+)`", r"\1", stripped)
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
    stripped = re.sub(r"\*\*([^*]+)\*\*", r"\1", stripped)
    stripped = re.sub(r"\*([^*]+)\*", r"\1", stripped)
    return stripped.strip()


def _description_candidate_score(text: str) -> int:
    lowered = text.lower()
    score = min(len(text), 240)
    if 40 <= len(text) <= 260:
        score += 80
    if any(word in lowered for word in ("use when", "use this", "helps", "provides", "guide", "workflow")):
        score += 30
    if lowered.startswith(("use when", "use this", "this skill", "helps", "provides")):
        score += 30
    if any(word in lowered for word in ("install", "usage", "example", "run ", "```")):
        score -= 30
    return score


def _is_meaningful_description(text: str) -> bool:
    if len(text) < 12:
        return False
    if len(text.split()) < 3:
        return False
    lowered = text.lower()
    if lowered.startswith(("usage:", "example:", "install:", "todo:")):
        return False
    return True


def _compact_description(text: str, *, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    boundary = max(text.rfind(".", 0, limit), text.rfind(";", 0, limit), text.rfind(",", 0, limit))
    if boundary >= 80:
        return text[: boundary + 1].strip()
    return text[:limit].rstrip()
