"""Line protocol for stdin-driven background agent workers."""

from __future__ import annotations

import json


def encode_worker_message(data: str) -> bytes:
    """Encode one worker message as exactly one UTF-8 line.

    Single-line text stays plain for compatibility with simple command
    overrides.  Multi-line text is wrapped as ``{"text": ...}`` so a
    readline-based worker does not split one logical prompt into multiple
    turns.
    """
    stripped = data.rstrip("\n")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        framed = stripped
    elif "\n" not in stripped and "\r" not in stripped:
        framed = stripped
    else:
        framed = json.dumps({"text": stripped}, ensure_ascii=False)
    return (framed + "\n").encode("utf-8")


def decode_worker_line(line: str) -> str:
    """Decode one worker protocol line into prompt text."""
    stripped = line.rstrip("\n")
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict) and isinstance(payload.get("text"), str):
        return payload["text"]
    return stripped
