"""YAML frontmatter parser for SKILL.md files.

A SKILL.md file looks like::

    ---
    name: my-skill
    description: Does something useful.
    model-invocable: true
    ---

    # my-skill

    Actual markdown content here...

Frontmatter is optional.  If absent, the name and description are
extracted from the first ``# Heading`` and first body paragraph.
"""

from __future__ import annotations

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

    # 1. Try to extract YAML frontmatter (between --- lines).
    stripped = content.lstrip("\n")
    if stripped.startswith("---\n"):
        parts = stripped.split("\n---\n", 1)
        if len(parts) >= 2:
            fm = _parse_yaml_safe(parts[0][4:])  # skip leading "---\n"
            body = parts[1] if len(parts) > 1 else ""

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
        "description": description.strip()[:200],
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


def _parse_yaml_safe(text: str) -> dict[str, Any]:
    """Parse YAML frontmatter.  Falls back to manual key:value parsing."""
    try:
        import yaml
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        pass

    # Manual fallback: parse simple "key: value" lines.
    result: dict[str, Any] = {}
    for line in text.split("\n"):
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower().replace("-", "_")
            val = val.strip().strip('"').strip("'")
            if key and val:
                result[key] = val
    return result


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
    """Extract description from the first content paragraph in the body."""
    lines = body.split("\n")
    heading_seen = False
    for line in lines:
        stripped = line.strip()
        # Skip blank lines and frontmatter delimiters.
        if stripped == "" or stripped == "---":
            continue
        # Track headings but don't use them as descriptions.
        if stripped.startswith("#"):
            heading_seen = True
            continue
        # First non-blank, non-heading line after any heading is the description.
        return stripped[:200]
    return ""
