"""Production-grade permission system tests — including P1 can_read bypass fix.

These tests verify that the permission system correctly handles:
- User-configured path deny rules for BOTH reads and writes
- Critical path protection as defense-in-depth
- Mode-based decisions
- Interactive confirmation flow
"""

from pathlib import Path
from miniharness.permissions import PermissionChecker, PermissionDecision, _ask_confirmation


# ===================================================================
# P1: can_read MUST respect user-configured deny rules.
#     The old code had a bypass: can_read would override any denial
#     unless the reason contained "sensitive".  Fixed.
# ===================================================================

def test_can_read_respects_user_deny_rules():
    """P1 fix: user path_rules work for reads, not just writes."""
    pc = PermissionChecker(
        cwd=Path("/home/user/project"),
        path_rules=[("/etc/*", False), ("/var/log/*", False)],
    )

    # Write: /etc/hostname should be denied by user rule.
    result = pc.can_write(Path("/etc/hostname"))
    assert not result.allowed, "Write to /etc should be denied by user rule"

    # Read: /etc/hostname should ALSO be denied by user rule.
    # (This was the old bug — can_read used to return PermissionDecision(True) for this.)
    result = pc.can_read(Path("/etc/hostname"))
    assert not result.allowed, (
        "P1 BUG: can_read should deny /etc access when user path_rules say so"
    )

    # Read: files NOT matching deny rules should still be allowed.
    result = pc.can_read(Path("/home/user/project/src/main.py"))
    assert result.allowed, "Read of project file should be allowed"

    print("P1 fix verified: can_read respects user deny rules.")


# ===================================================================
# Critical path defense-in-depth — unoverridable by any mode
# ===================================================================

def test_critical_paths_always_denied():
    """Critical paths (SSH, cloud creds, /etc/shadow) are ALWAYS denied."""
    # Even in bypass mode, critical paths are blocked.
    pc = PermissionChecker(cwd=Path("/tmp"), mode="bypass")

    critical_cases = [
        (Path.home() / ".ssh/id_rsa", True),       # should deny
        (Path.home() / ".ssh/id_ed25519", True),    # should deny
        (Path.home() / ".ssh/known_hosts", False),   # should allow (not a key)
        (Path("/etc/shadow"), True),                 # should deny
        (Path("/etc/hostname"), False),              # should allow
        (Path.home() / ".aws/credentials", True),    # should deny
    ]

    for path, should_deny in critical_cases:
        result = pc.can_read(path)
        if should_deny:
            assert not result.allowed, (
                f"Critical path {path} should be denied even in bypass mode"
            )
        else:
            assert result.allowed, (
                f"Non-critical path {path} should be allowed"
            )

    print("Critical path protection verified.")


# ===================================================================
# Mode behavior
# ===================================================================

def test_modes():
    cwd = Path("/tmp/test")
    test_file = cwd / "test.py"

    # bypass: everything allowed (except critical paths).
    pc = PermissionChecker(cwd=cwd, mode="bypass")
    assert pc.can_write(test_file).allowed
    # In bypass mode, commands are auto-allowed via evaluate() directly
    # (can_run_command would trigger interactive prompt).
    assert pc.evaluate(tool_name="bash", command="ls").allowed

    # plan: read-only.
    pc = PermissionChecker(cwd=cwd, mode="plan")
    assert pc.can_read(test_file).allowed
    assert not pc.evaluate(tool_name="write_file", file_path=str(test_file)).allowed
    assert not pc.evaluate(tool_name="bash", command="ls").allowed

    # accept-edits: writes auto-allowed, commands require confirmation.
    pc = PermissionChecker(cwd=cwd, mode="accept-edits")
    assert pc.evaluate(tool_name="write_file", file_path=str(test_file)).allowed
    cmd_result = pc.evaluate(tool_name="bash", command="ls")
    assert cmd_result.requires_confirmation
    assert "Shell commands require confirmation" in cmd_result.reason

    # default: all mutations require confirmation.
    pc = PermissionChecker(cwd=cwd, mode="default")
    write_result = pc.evaluate(tool_name="write_file", file_path=str(test_file))
    assert write_result.requires_confirmation
    cmd_result = pc.evaluate(tool_name="bash", command="ls")
    assert cmd_result.requires_confirmation

    print("Mode behavior verified.")


def test_confirmation_accepts_single_yes_line(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "y")

    assert _ask_confirmation("Allow test?") is True


def test_confirmation_denies_empty_line(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda: "")

    assert _ask_confirmation("Allow test?") is False


# ===================================================================
# Tool allow/deny lists
# ===================================================================

def test_tool_lists():
    pc = PermissionChecker(
        cwd=Path("/tmp"),
        allowed_tools=["read_file", "ls", "grep"],
        denied_tools=["bash"],
    )

    # Allowed tools pass.
    assert pc.evaluate(tool_name="read_file", is_read_only=True).allowed

    # Denied tool is blocked.
    result = pc.evaluate(tool_name="bash", command="ls")
    assert not result.allowed

    # Tool not in allow list is blocked.
    result = pc.evaluate(tool_name="write_file", file_path="/tmp/x.txt")
    assert not result.allowed

    print("Tool allow/deny lists verified.")


# ===================================================================
# User-configured command deny patterns
# ===================================================================

def test_command_deny_patterns():
    pc = PermissionChecker(
        cwd=Path("/tmp"),
        denied_commands=["pip install*", "npm install -g*"],
    )

    # Matching commands denied.
    assert not pc.evaluate(tool_name="bash", command="pip install requests").allowed
    assert not pc.evaluate(tool_name="bash", command="npm install -g react").allowed

    # Non-matching allowed (subject to mode).
    assert pc.evaluate(tool_name="bash", command="ls -la").requires_confirmation

    print("Command deny patterns verified.")


# ===================================================================
# PermissionDecision dataclass
# ===================================================================

def test_permission_decision():
    allowed = PermissionDecision(True)
    assert allowed.allowed
    assert not allowed.requires_confirmation
    assert allowed.reason == ""

    blocked = PermissionDecision(False, reason="denied by policy")
    assert not blocked.allowed
    assert blocked.reason == "denied by policy"

    confirm = PermissionDecision(False, requires_confirmation=True, reason="ask user")
    assert not confirm.allowed
    assert confirm.requires_confirmation

    print("PermissionDecision dataclass verified.")


# ===================================================================
# Mode cycling
# ===================================================================

def test_mode_cycle():
    pc = PermissionChecker(cwd=Path("/tmp"), mode="default")
    assert pc.mode == "default"
    assert pc.cycle_mode() == "accept-edits"
    assert pc.cycle_mode() == "bypass"
    assert pc.cycle_mode() == "plan"
    assert pc.cycle_mode() == "default"  # wraps around
    print("Mode cycling verified.")


# ===================================================================
# Relative path helper
# ===================================================================

def test_relative_path():
    pc = PermissionChecker(cwd=Path("/home/user/project"))
    assert pc._relative(Path("/home/user/project/src/main.py")) == "src/main.py"
    assert pc._relative(Path("/etc/passwd")) == "/etc/passwd"  # outside cwd
    print("Relative path helper verified.")


# ===================================================================
if __name__ == "__main__":
    test_can_read_respects_user_deny_rules()
    test_critical_paths_always_denied()
    test_modes()
    test_tool_lists()
    test_command_deny_patterns()
    test_permission_decision()
    test_mode_cycle()
    test_relative_path()
    print()
    print("=== ALL 8 permission system tests passed! ===")
