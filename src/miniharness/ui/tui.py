"""TUI frontend — Textual app for MiniHarness.

Spawns ``--backend-only`` as a subprocess, reads ``MHJSON:`` events from
stdout, renders them in a clean terminal UI.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Header, Input, Label, Static
from textual.screen import ModalScreen
from textual import events

from miniharness.ui.protocol import decode_event


# ═══════════════════════════════════════════════════════════════════════════
# Permission Modal — clean, minimal
# ═══════════════════════════════════════════════════════════════════════════


class PermissionModal(ModalScreen[bool]):
    """Prompt the user for permission (y/n)."""

    DEFAULT_CSS = """
    PermissionModal {
        align: center middle;
    }
    #perm-box {
        width: 56;
        height: auto;
        padding: 1 2;
        border: solid grey;
        background: black;
    }
    #perm-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    #perm-body {
        width: 100%;
        margin-bottom: 1;
    }
    #perm-hint {
        width: 100%;
        text-align: center;
        color: grey;
        text-style: italic;
    }
    """

    def __init__(self, tool_name: str, prompt: str) -> None:
        super().__init__()
        self.tool_name = tool_name
        self.prompt = prompt

    def compose(self) -> ComposeResult:
        with Container(id="perm-box"):
            yield Label(f"Allow {self.tool_name}?", id="perm-title")
            yield Label(self.prompt[:200], id="perm-body")
            yield Label("[Y] Allow  [N] Deny", id="perm-hint")

    def on_key(self, event: events.Key) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key == "n" or event.key == "escape":
            self.dismiss(False)


# ═══════════════════════════════════════════════════════════════════════════
# Main App
# ═══════════════════════════════════════════════════════════════════════════


class MiniHarnessTUI(App):
    """Textual TUI — clean, minimal, mirrors Claude Code layout."""

    CSS = """
    #conversation {
        height: 1fr;
        padding: 0 1;
    }
    .assistant-text {
        color: green;
    }
    .tool-line {
        color: grey;
    }
    .tool-error {
        color: red;
    }
    .user-line {
        color: white;
        text-style: bold;
    }
    .system-line {
        color: grey;
        text-style: italic;
    }
    #prompt-input {
        dock: bottom;
        margin: 0 0 1 0;
    }
    #status-line {
        dock: bottom;
        height: 1;
        color: grey;
        text-style: italic;
    }
    """

    BINDINGS = [
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        *,
        cwd: Path,
        prompt: str | None = None,
        resume_session_id: str | None = None,
    ) -> None:
        super().__init__()
        self.cwd = cwd
        self._initial_prompt = prompt
        self._resume_id = resume_session_id
        self._proc: subprocess.Popen | None = None
        self._reader_task: asyncio.Task | None = None
        self._streaming_widget: Static | None = None
        self._streamed_text: str = ""   # fully displayed text on the widget
        self._pending_text: str = ""    # new deltas not yet pushed to widget
        self._flush_timer: asyncio.Task | None = None
        self._busy: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Spawn backend, begin reading events."""
        cmd = [
            sys.executable, "-m", "miniharness", "--backend-only",
            "--cwd", str(self.cwd),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self._reader_task = asyncio.create_task(self._read_events())

        startup_reqs = self._startup_requests()
        if startup_reqs:
            self._busy = True

        for req in startup_reqs:
            self._send(req)

        self.query_one("#prompt-input", Input).focus()

    def on_unmount(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
        if self._proc:
            try:
                self._send({"type": "shutdown"})
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield VerticalScroll(id="conversation")
        yield Static("", id="status-line")
        yield Input(placeholder="▸ Type a prompt or /command...", id="prompt-input")

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if self._busy:
            self._status("Busy — wait for the current turn to finish.")
            event.input.value = ""
            return

        line = event.value.strip()
        event.input.value = ""
        if not line:
            return
        if line in ("/exit", "/quit", "/q"):
            self.exit()
            return

        self._busy = True
        self._add(f"▸ {line}", "user-line")
        self._send({"type": "submit_line", "line": line})

    def action_quit(self) -> None:
        self.exit()

    def _startup_requests(self) -> list[dict]:
        """Build initial requests to send on mount."""
        reqs: list[dict] = []
        if self._resume_id:
            line = "/resume" if self._resume_id == "latest" else f"/resume {self._resume_id}"
            reqs.append({"type": "submit_line", "line": line})
        if self._initial_prompt:
            reqs.append({"type": "submit_line", "line": self._initial_prompt})
        return reqs

    # ------------------------------------------------------------------
    # Read MHJSON: from backend stdout
    # ------------------------------------------------------------------

    async def _read_events(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        loop = asyncio.get_event_loop()
        while True:
            try:
                raw = await loop.run_in_executor(None, self._proc.stdout.readline)
            except Exception:
                break
            if not raw:
                break
            line = raw.strip()
            evt = decode_event(line)
            if evt is None:
                if line:
                    self._add(f"[backend] {line}", "system-line")
                continue
            self._dispatch(evt)

    # ------------------------------------------------------------------
    # Event dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, evt: dict) -> None:
        t = evt.get("type", "")

        if t == "ready":
            m = evt.get("model", "?")
            s = (evt.get("session_id", "") or "")[:12]
            self._status(f"Model: {m}  |  Session: {s}")

        elif t == "assistant_delta":
            self._pending_text += evt.get("text", "")

            if self._streaming_widget is None:
                self._streaming_widget = Static("", classes="assistant-text")
                self.query_one("#conversation", VerticalScroll).mount(self._streaming_widget)

            if self._flush_timer is None:
                self._schedule_flush()

        elif t == "assistant_complete":
            self._flush_now(reschedule=False)
            self._streaming_widget = None
            self._streamed_text = ""
            self._pending_text = ""
            self._clear_busy()

        elif t == "tool_started":
            name = evt.get("tool_name", "?")
            inp = evt.get("tool_input", {})
            s = json.dumps(inp, ensure_ascii=False)[:120]
            self._add(f"  → {name}({s})", "tool-line")

        elif t == "tool_completed":
            out = (evt.get("output", "") or "")[:120].replace("\n", " ")
            cls = "tool-error" if evt.get("is_error") else "tool-line"
            self._add(f"  ← {out}", cls)

        elif t == "permission_request":
            rid = evt.get("request_id", "")
            tn = evt.get("tool_name", "")
            p = evt.get("prompt", "")

            def _done(allowed: bool) -> None:
                self._send({"type": "permission_response", "request_id": rid, "allowed": allowed})
                self._status(f"{tn}: {'allowed' if allowed else 'denied'}")

            self.push_screen(PermissionModal(tn, p), _done)

        elif t == "system_message":
            # Multiline system output (e.g., /help) → show in conversation.
            msg = evt.get("message", "") or ""
            self._add(msg, "system-line")

        elif t == "status":
            self._status((evt.get("message", "") or "")[:120])

        elif t == "error":
            self._add(f"Error: {evt.get('message', '')}", "tool-error")
            self._clear_busy()

        elif t == "shutdown":
            self._add("Backend shut down.", "system-line")
            self._clear_busy()

        elif t == "line_complete":
            self._clear_busy()

    # ------------------------------------------------------------------
    # Streaming flush (80ms timer, like OpenHarness's double-buffer)
    # ------------------------------------------------------------------

    def _schedule_flush(self) -> None:
        """Schedule a flush of pending assistant text after 80ms."""
        async def _flush_after_delay() -> None:
            await asyncio.sleep(0.08)
            self._flush_now()

        self._flush_timer = asyncio.create_task(_flush_after_delay())

    def _flush_now(self, *, reschedule: bool = True) -> None:
        """Push pending deltas to the streaming widget."""
        if self._flush_timer is not None:
            self._flush_timer.cancel()
            self._flush_timer = None

        if self._pending_text and self._streaming_widget is not None:
            self._streamed_text += self._pending_text
            try:
                self._streaming_widget.update(self._streamed_text)
                self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)
            except Exception as exc:
                print(f"TUI render error: {exc}", file=sys.stderr)

        self._pending_text = ""

        if reschedule and self._streaming_widget is not None:
            self._schedule_flush()

    # ------------------------------------------------------------------
    # Busy state
    # ------------------------------------------------------------------

    def _clear_busy(self) -> None:
        self._busy = False
        try:
            self.query_one("#prompt-input", Input).focus()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add(self, text: str, css_class: str) -> None:
        try:
            self.query_one("#conversation", VerticalScroll).mount(
                Label(text, classes=css_class) if css_class else Label(text)
            )
            self.query_one("#conversation", VerticalScroll).scroll_end(animate=False)
        except Exception as exc:
        # during development
            print(f"TUI render error: {exc}", file=sys.stderr)

    def _status(self, text: str) -> None:
        try:
            self.query_one("#status-line", Static).update(text[:120])
        except Exception as exc:
            # during development
            print(f"TUI render error: {exc}", file=sys.stderr)
            pass

    def _send(self, msg: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            return
        try:
            self._proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════


def run_tui(*, cwd: Path, prompt: str | None = None, resume_session_id: str | None = None) -> None:
    MiniHarnessTUI(cwd=cwd, prompt=prompt, resume_session_id=resume_session_id).run()
