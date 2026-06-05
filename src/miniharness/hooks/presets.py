"""Production hook presets — configurable, not hardcoded.

Each preset is a function that returns a dict suitable for
``load_hook_registry()``.  Users can:

1. Enable/disable individual presets.
2. Extend or override the pattern lists.
3. Combine presets with custom hooks.

Design principle: the *patterns* are data, not code.  They live in
well-documented lists so users can inspect, modify, or replace them
without touching the hook execution engine.

Usage::

    from miniharness.hooks.presets import (
        dangerous_command_preset,
        sensitive_file_preset,
        code_security_preset,
        audit_log_preset,
        approval_preset,
        production_presets,
    )

    # Enable individual presets:
    config = dangerous_command_preset()
    config.update(sensitive_file_preset())

    # Or enable all production presets at once:
    config = production_presets()
"""

from __future__ import annotations

from typing import Any

# ═══════════════════════════════════════════════════════════════════════════
# 1. Dangerous Commands — patterns that should ALWAYS trigger a block
# ═══════════════════════════════════════════════════════════════════════════

# These are fnmatch patterns matched against the extracted subjects from
# _extract_matchable_subjects().  The "bash:*" prefix targets bash tool
# commands; "bash:" patterns match the raw command string.

DANGEROUS_COMMAND_PATTERNS: list[str] = [
    # ── Destructive deletion ──────────────────────────────────────
    "bash:*rm -rf*",
    "bash:*rm  -rf*",
    "bash:*rm   -rf*",
    "bash:*rm -r /*",
    "bash:*rm -rf /*",
    "bash:*rm -rf ~*",
    "bash:*rm -rf /home*",
    "bash:*find* -delete*",
    "bash:*:(){ :|:& };:*",        # fork bomb
    # ── Disk / filesystem destruction ─────────────────────────────
    "bash:*dd if=*",
    "bash:*mkfs*",
    "bash:*fdisk*",
    "bash:*> /dev/sd*",
    "bash:*dd of=/dev/*",
    # ── System control ────────────────────────────────────────────
    "bash:*shutdown*",
    "bash:*reboot*",
    "bash:*halt*",
    "bash:*poweroff*",
    "bash:*init 0*",
    "bash:*init 6*",
    # ── Permission escalation / backdoors ─────────────────────────
    "bash:*chmod 777 /*",
    "bash:*chmod -R 777*",
    "bash:*chown -R * /*",
    "bash:*curl*|*bash*",
    "bash:*wget*|*sh*",
    "bash:*eval*$(curl*",
    "bash:*eval*$(wget*",
    # ── Sensitive data exposure ───────────────────────────────────
    "bash:*cat /etc/shadow*",
    "bash:*cat /etc/passwd*",
    "bash:*cat ~/.ssh/id_rsa*",
    "bash:*cat ~/.aws/credentials*",
    "bash:*printenv*",
    "bash:*env | grep*",
    # ── Package / system tampering ────────────────────────────────
    "bash:*pip install*--break-system-packages*",
    "bash:*npm install -g*",
    "bash:*gem install*",
    "bash:*cargo install*",
    # ── Network / reverse shell ───────────────────────────────────
    "bash:*nc -e*",
    "bash:*ncat -e*",
    "bash:*bash -i >&*",
    "bash:*python -c*import socket*",
]


def dangerous_command_preset() -> dict[str, list[dict[str, Any]]]:
    """Pre-built hooks that BLOCK dangerous shell commands.

    Each pattern in ``DANGEROUS_COMMAND_PATTERNS`` gets its own hook
    so blocking is precise and auditable.
    """
    hooks: list[dict[str, Any]] = []
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        hooks.append({
            "type": "command",
            "command": (
                f"echo 'BLOCKED: $TOOL_NAME $COMMAND matched {pattern}' "
                f">>&2; exit 1"
            ),
            "matcher": pattern,
            "block_on_failure": True,
            "timeout_seconds": 5,
        })

    return {"pre_tool_use": hooks}


# ═══════════════════════════════════════════════════════════════════════════
# 2. Sensitive File Protection — paths that should never be written
# ═══════════════════════════════════════════════════════════════════════════

SENSITIVE_FILE_PATTERNS: list[str] = [
    # System configuration
    "write_file:/etc/*",
    "edit_file:/etc/*",
    "write_file:/boot/*",
    "edit_file:/boot/*",
    # SSH / credentials
    "write_file:*/.ssh/*",
    "edit_file:*/.ssh/*",
    "write_file:*/.aws/*",
    "edit_file:*/.aws/*",
    "write_file:*/.gcloud/*",
    "edit_file:*/.gcloud/*",
    "write_file:*/.azure/*",
    "edit_file:*/.azure/*",
    "write_file:*/.gnupg/*",
    "edit_file:*/.gnupg/*",
    # Container / orchestration configs
    "write_file:*/.docker/*",
    "edit_file:*/.docker/*",
    "write_file:*/.kube/*",
    "edit_file:*/.kube/*",
    # Git / project config (often contains secrets)
    "write_file:*/.git/config",
    "edit_file:*/.git/config",
    "write_file:*/.git-credentials",
    "edit_file:*/.git-credentials",
    "write_file:*/.env.production*",
    "edit_file:*/.env.production*",
    "write_file:*/.env.local*",
    "edit_file:*/.env.local*",
    # System / root
    "write_file:/root/*",
    "edit_file:/root/*",
    "write_file:/usr/*",
    "edit_file:/usr/*",
    "write_file:/var/*",
    "edit_file:/var/*",
    # Harness own credential stores
    "write_file:*/.miniharness/credentials*",
    "edit_file:*/.miniharness/credentials*",
    "write_file:*/.openharness/credentials*",
    "edit_file:*/.openharness/credentials*",
]


def sensitive_file_preset() -> dict[str, list[dict[str, Any]]]:
    """Pre-built hooks that BLOCK writes to sensitive file paths."""
    hooks: list[dict[str, Any]] = []
    for pattern in SENSITIVE_FILE_PATTERNS:
        hooks.append({
            "type": "command",
            "command": (
                f"echo 'BLOCKED: $TOOL_NAME attempted to write to "
                f"$PATH (matched {pattern})' >&2; exit 1"
            ),
            "matcher": pattern,
            "block_on_failure": True,
            "timeout_seconds": 5,
        })

    return {"pre_tool_use": hooks}


# ═══════════════════════════════════════════════════════════════════════════
# 3. Code Security Review — LLM-powered pattern detection
# ═══════════════════════════════════════════════════════════════════════════

_CODE_SECURITY_PROMPT = """\
You are a code security reviewer.  Analyze the following tool call and
determine if it could introduce a security vulnerability.

Payload: $ARGUMENTS

Check for:
- Hardcoded secrets (API keys, passwords, tokens, private keys)
- SQL injection (string concatenation in queries)
- Command injection (os.system with user input, shell=True without sanitization)
- Path traversal (using user input directly in file paths)
- Insecure deserialization (pickle.loads on untrusted data)
- Weak cryptography (MD5, SHA1 for passwords, hardcoded salts)

Return ONLY a JSON object:
{"ok": true} if the code looks safe.
{"ok": false, "reason": "describe the specific vulnerability found"} if unsafe.

Do NOT include any text outside the JSON object."""


def code_security_preset() -> dict[str, list[dict[str, Any]]]:
    """Pre-built hook that uses an LLM to review generated code for
    security vulnerabilities before it's written to disk.

    This is a prompt-type hook — it asks the model to evaluate the
    code.  Requires ``llm_stream`` in ``HookExecutionContext``.
    """
    return {
        "pre_tool_use": [
            {
                "type": "prompt",
                "prompt": _CODE_SECURITY_PROMPT,
                "matcher": "write_file:*",
                "block_on_failure": True,
                "timeout_seconds": 60,
            },
            {
                "type": "prompt",
                "prompt": _CODE_SECURITY_PROMPT,
                "matcher": "edit_file:*",
                "block_on_failure": True,
                "timeout_seconds": 60,
            },
            # Also review pip/npm/cargo install for typo-squatting risks.
            {
                "type": "prompt",
                "prompt": (
                    "Check if this package install command could install "
                    "malicious or typo-squatted packages. Payload: $ARGUMENTS. "
                    'Return {"ok": true} or {"ok": false, "reason": "..."}.'
                ),
                "matcher": "bash:*pip install*",
                "block_on_failure": False,  # warn but don't block
                "timeout_seconds": 30,
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Audit Logging — record ALL agent actions
# ═══════════════════════════════════════════════════════════════════════════

def audit_log_preset(
    *,
    log_dir: str = "~/.miniharness/audit",
) -> dict[str, list[dict[str, Any]]]:
    """Pre-built hooks that log ALL tool executions to audit files.

    Creates timestamped JSONL audit logs.  Never blocks — audit hooks
    are always ``block_on_failure: false``.

    Parameters
    ----------
    log_dir:
        Directory for audit log files (default: ~/.miniharness/audit).
    """
    # Expand ~ in log_dir.
    import os as _os
    log_dir = _os.path.expanduser(log_dir)

    return {
        # ── Log every tool execution (pre + post for full trace) ──
        "pre_tool_use": [
            {
                "type": "command",
                "command": (
                    f"mkdir -p {log_dir} && "
                    f"echo '{{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
                    f"\"event\":\"pre_tool_use\","
                    f"\"tool\":\"$TOOL_NAME\","
                    f"\"session\":\"$SESSION_ID\"}}' "
                    f">> {log_dir}/audit.jsonl"
                ),
                "matcher": None,
                "block_on_failure": False,
                "timeout_seconds": 5,
            },
        ],
        "post_tool_use": [
            {
                "type": "command",
                "command": (
                    f"echo '{{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
                    f"\"event\":\"post_tool_use\","
                    f"\"tool\":\"$TOOL_NAME\","
                    f"\"session\":\"$SESSION_ID\","
                    f"\"is_error\":$IS_ERROR}}' "
                    f">> {log_dir}/audit.jsonl"
                ),
                "matcher": None,
                "block_on_failure": False,
                "timeout_seconds": 10,
            },
        ],
        "tool_failed": [
            {
                "type": "command",
                "command": (
                    f"echo '{{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
                    f"\"event\":\"tool_failed\","
                    f"\"tool\":\"$TOOL_NAME\","
                    f"\"error\":\"$ERROR\","
                    f"\"session\":\"$SESSION_ID\"}}' "
                    f">> {log_dir}/failures.jsonl"
                ),
                "matcher": None,
                "block_on_failure": False,
                "timeout_seconds": 10,
            },
        ],
        # ── Session start marker ──────────────────────────────────
        "session_start": [
            {
                "type": "command",
                "command": (
                    f"mkdir -p {log_dir} && "
                    f"echo '{{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
                    f"\"event\":\"session_start\","
                    f"\"model\":\"$MODEL\","
                    f"\"cwd\":\"$CWD\","
                    f"\"session\":\"$SESSION_ID\"}}' "
                    f">> {log_dir}/audit.jsonl"
                ),
                "matcher": None,
                "block_on_failure": False,
                "timeout_seconds": 10,
            },
        ],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Human Approval — interactive confirmation for high-risk operations
# ═══════════════════════════════════════════════════════════════════════════

APPROVAL_PATTERNS: list[str] = [
    "bash:*rm -rf*",
    "bash:*rm  -rf*",
    "bash:*git push --force*",
    "bash:*git push -f*",
    "bash:*docker rm*",
    "bash:*docker system prune*",
    "bash:*kubectl delete*",
    "bash:*kubectl apply*",
    "bash:*terraform apply*",
    "bash:*terraform destroy*",
    "bash:*aws * delete*",
    "bash:*gcloud * delete*",
    "bash:*heroku * destroy*",
    "bash:*flyctl * destroy*",
    "bash:*DROP DATABASE*",
    "bash:*DROP TABLE*",
    "bash:*TRUNCATE*",
    "write_file:*/.env.production*",
    "edit_file:*/.env.production*",
    "bash:*python setup.py*",
    "bash:*pip install*",
    "bash:*npm publish*",
    "bash:*cargo publish*",
    "bash:*docker push*",
]


def approval_preset() -> dict[str, list[dict[str, Any]]]:
    """Pre-built hooks that REQUIRE HUMAN APPROVAL for high-risk operations.

    Uses the ``confirm`` hook type — the user must type ``yes`` at an
    interactive prompt.  Timeout or any other response denies the action.
    """
    hooks: list[dict[str, Any]] = []
    for pattern in APPROVAL_PATTERNS:
        # Extract a human-readable label from the pattern.
        label = pattern.replace("bash:", "").replace("write_file:", "").replace("edit_file:", "")
        # Truncate glob patterns for display.
        if len(label) > 60:
            label = label[:57] + "..."

        hooks.append({
            "type": "confirm",
            "message": (
                f"High-risk operation detected: {label}\n"
                f"Tool: $TOOL_NAME\n"
                f"Session: $SESSION_ID\n\n"
                f"Command: $COMMAND"
            ),
            "matcher": pattern,
            "timeout_seconds": 300,  # 5 minutes to respond
        })

    return {"pre_tool_use": hooks}


# ═══════════════════════════════════════════════════════════════════════════
# 6. Production bundle — all presets combined
# ═══════════════════════════════════════════════════════════════════════════

def production_presets(
    *,
    audit_log_dir: str = "~/.miniharness/audit",
    enable_code_review: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Combine all production presets into a single configuration dict.

    This is the recommended starting point for a production deployment.
    Individual presets can be enabled/disabled via parameters.

    Parameters
    ----------
    audit_log_dir:
        Directory for audit log files.
    enable_code_review:
        If False, skip the LLM-based code security review (saves API
        costs if you're using a separate security scanner).

    Returns
    -------
    dict
        Ready for ``load_hook_registry()`` or ``Settings.hooks``.
    """
    config: dict[str, list[dict[str, Any]]] = {}

    # Always-on: safety presets.
    merge_config(config, dangerous_command_preset())
    merge_config(config, sensitive_file_preset())
    merge_config(config, approval_preset())

    # Always-on: audit logging.
    merge_config(config, audit_log_preset(log_dir=audit_log_dir))

    # Optional: code security review (uses LLM tokens).
    if enable_code_review:
        merge_config(config, code_security_preset())

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def merge_config(
    base: dict[str, list[dict[str, Any]]],
    overlay: dict[str, list[dict[str, Any]]],
) -> None:
    """Merge *overlay* into *base*, combining hook lists per event."""
    for event, hooks in overlay.items():
        base.setdefault(event, []).extend(hooks)
