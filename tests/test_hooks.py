"""Integration test for the MiniHarness Hook system."""
import asyncio, tempfile, os
from pathlib import Path

import pytest

from miniharness.hooks import (
    HookEvent, HookRegistry, HookExecutor, HookExecutionContext,
    HookResult, AggregatedHookResult, load_hook_registry,
)
from miniharness.hooks.executor import (
    _matches_hook, _parse_hook_response, _inject_arguments,
    _extract_matchable_subjects,
)
from miniharness.hooks.schemas import CommandHookDefinition


# ===================================================================
# Test 1: load_hook_registry
# ===================================================================
def test_registry():
    config = {
        "pre_tool_use": [
            {"type": "command", "command": "echo audit", "matcher": "bash"},
        ],
        "session_start": [
            {"type": "command", "command": "echo started"},
        ],
        "post_tool_use": [
            {"type": "command", "command": "echo done"},
            {"type": "prompt", "prompt": "Is this safe?",
             "matcher": "rm*", "block_on_failure": True},
            {"type": "command", "command": "echo final"},
        ],
    }
    registry = load_hook_registry(config)
    assert registry.get(HookEvent.PRE_TOOL_USE)[0].matcher == "bash"
    assert len(registry.get(HookEvent.POST_TOOL_USE)) == 3
    assert registry.total_count == 5
    print("1. Registry: OK (5 hooks)")

    # Invalid events silently skipped.
    empty = load_hook_registry({"bad_event": [{"type": "command", "command": "x"}]})
    assert empty.total_count == 0
    print("2. Invalid event skipped: OK")


# ===================================================================
# Test 2: _matches_hook (basic — tool_name only)
# ===================================================================
def test_matcher_basic():
    h = CommandHookDefinition(command="echo x", matcher="bash")
    assert _matches_hook(h, {"tool_name": "bash"}) is True
    assert _matches_hook(h, {"tool_name": "read_file"}) is False

    # No matcher → always matches.
    h2 = CommandHookDefinition(command="echo x")
    assert _matches_hook(h2, {"tool_name": "anything"}) is True

    print("3. Matcher (basic): OK")


# ===================================================================
# Test 2b: _matches_hook (tool_input matching — the key fix!)
# ===================================================================
def test_matcher_tool_input():
    # matcher="bash:rm*" — matches bash tool whose command starts with "rm"
    h = CommandHookDefinition(command="echo blocked", matcher="bash:rm*")

    # bash tool calling rm → MATCH
    payload = {
        "tool_name": "bash",
        "tool_input": {"command": "rm -rf /tmp/cache"},
    }
    assert _matches_hook(h, payload) is True

    # bash tool calling ls → NO MATCH
    payload2 = {
        "tool_name": "bash",
        "tool_input": {"command": "ls -la"},
    }
    assert _matches_hook(h, payload2) is False

    # read_file has no "command" field, and tool_name="read_file" ≠ "bash:..."
    # → NO MATCH
    payload3 = {
        "tool_name": "read_file",
        "tool_input": {"path": "/tmp/test.txt"},
    }
    assert _matches_hook(h, payload3) is False

    print("4. Matcher (tool_input): OK")


# ===================================================================
# Test 2c: _matches_hook (cross-tool patterns)
# ===================================================================
def test_matcher_cross_tool():
    # matcher="*:pip install*" — any tool with "pip install" in any input
    h = CommandHookDefinition(command="echo blocked", matcher="*:pip install*")

    # bash tool → MATCH
    assert _matches_hook(h, {
        "tool_name": "bash",
        "tool_input": {"command": "pip install requests"},
    }) is True

    # write_file with no matching input → NO MATCH
    assert _matches_hook(h, {
        "tool_name": "write_file",
        "tool_input": {"path": "setup.py", "content": "..."},
    }) is False

    # read_file on /etc path.
    h2 = CommandHookDefinition(command="echo blocked", matcher="read_file:/etc/*")
    assert _matches_hook(h2, {
        "tool_name": "read_file",
        "tool_input": {"path": "/etc/passwd"},
    }) is True
    assert _matches_hook(h2, {
        "tool_name": "read_file",
        "tool_input": {"path": "/home/user/test.txt"},
    }) is False

    print("5. Matcher (cross-tool): OK")


# ===================================================================
# Test 2d: _extract_matchable_subjects
# ===================================================================
def test_extract_subjects():
    # Bash command payload.
    subjects = _extract_matchable_subjects({
        "tool_name": "bash",
        "tool_input": {"command": "rm -rf /tmp"},
        "event": "pre_tool_use",
    })
    assert "bash" in subjects
    assert "bash:rm -rf /tmp" in subjects
    assert "rm -rf /tmp" in subjects
    assert "pre_tool_use" in subjects

    # File tool payload.
    subjects = _extract_matchable_subjects({
        "tool_name": "write_file",
        "tool_input": {"path": "/etc/hostname", "content": "myhost"},
        "event": "pre_tool_use",
    })
    assert "write_file" in subjects
    assert "write_file:/etc/hostname" in subjects
    assert "/etc/hostname" in subjects

    print("6. Extracted subjects: OK")


# ===================================================================
# Test 2e: _matches_hook (glob patterns — original test)
# ===================================================================
def test_matcher_glob():
    h = CommandHookDefinition(command="echo x", matcher="rm*")
    # With the new multi-subject matching, "rm*" matches "rm -rf /" as a
    # raw tool_input value subject
    assert _matches_hook(h, {
        "tool_name": "bash",
        "tool_input": {"command": "rm -rf /"},
    }) is True

    assert _matches_hook(h, {
        "tool_name": "bash",
        "tool_input": {"command": "ls"},
    }) is False

    print("7. Matcher (glob): OK")


# ===================================================================
# Test 3: _parse_hook_response
# ===================================================================
def test_parser():
    # Strict JSON.
    assert _parse_hook_response('{"ok": true}') == {"ok": True}
    assert _parse_hook_response('{"ok": false, "reason": "bad"}') == {"ok": False, "reason": "bad"}

    # Simple affirmative.
    assert _parse_hook_response("OK") == {"ok": True}
    assert _parse_hook_response("Yes") == {"ok": True}
    assert _parse_hook_response("safe") == {"ok": True}

    # Negative → uses text as reason.
    result = _parse_hook_response("This looks dangerous")
    assert result["ok"] is False
    assert "dangerous" in result["reason"]

    # Code-fenced JSON.
    fenced = '```json\n{"ok": true}\n```'
    assert _parse_hook_response(fenced) == {"ok": True}

    # Empty string → treated as invalid response.
    result = _parse_hook_response("")
    assert result["ok"] is False

    print("4. Response parser: OK")


# ===================================================================
# Test 4: _inject_arguments
# ===================================================================
def test_injection():
    r = _inject_arguments("echo $TOOL_NAME", {"tool_name": "bash"})
    assert "bash" in r

    r2 = _inject_arguments("payload=$ARGUMENTS", {"x": "y"})
    assert "x" in r2 or '"x"' in r2

    # Shell escape wraps values in quotes.
    r3 = _inject_arguments("echo $TOOL_NAME", {"tool_name": "rm -rf"},
                           shell_escape=True)
    assert "rm -rf" in r3

    print("5. Argument injection: OK")


# ===================================================================
# Test 5: Command hook (real subprocess)
# ===================================================================
@pytest.mark.asyncio
async def test_command_hook():
    tmp = Path(tempfile.mktemp(suffix=".log"))
    reg = load_hook_registry({
        "pre_tool_use": [
            {"type": "command", "command": f"echo ok > {tmp}"},
        ],
    })
    ctx = HookExecutionContext(cwd=Path.cwd())
    ex = HookExecutor(reg, ctx)
    r = await ex.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert r.all_passed and not r.blocked
    assert tmp.exists()
    content = tmp.read_text().strip()
    assert "ok" in content
    tmp.unlink(missing_ok=True)
    print("6. Command hook: OK")


# ===================================================================
# Test 6: Blocking hook
# ===================================================================
@pytest.mark.asyncio
async def test_blocking():
    reg = load_hook_registry({
        "pre_tool_use": [
            {"type": "command", "command": "exit 1", "block_on_failure": True},
        ],
    })
    ex = HookExecutor(reg, HookExecutionContext(cwd=Path.cwd()))
    r = await ex.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert r.blocked
    assert len(r.results) == 1
    assert r.results[0].success is False
    print(f"7. Blocking hook: OK (reason: {r.reason[:50]})")


# ===================================================================
# Test 7: Runtime matcher filtering
# ===================================================================
@pytest.mark.asyncio
async def test_runtime_matcher():
    reg = load_hook_registry({
        "pre_tool_use": [
            {"type": "command", "command": "echo x", "matcher": "bash"},
        ],
    })
    ex = HookExecutor(reg, HookExecutionContext(cwd=Path.cwd()))
    r1 = await ex.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "bash"})
    assert len(r1.results) == 1
    r2 = await ex.execute(HookEvent.PRE_TOOL_USE, {"tool_name": "read_file"})
    assert len(r2.results) == 0
    print("8. Runtime matcher: OK")


# ===================================================================
# Test 8: AggregatedHookResult
# ===================================================================
def test_aggregated():
    agg = AggregatedHookResult(results=[
        HookResult(hook_type="command", success=True, output="ok"),
        HookResult(hook_type="command", success=False, output="fail",
                   blocked=True, reason="exit 1"),
    ])
    assert agg.blocked is True
    assert agg.all_passed is False
    assert "exit 1" in agg.reason

    agg2 = AggregatedHookResult(results=[
        HookResult(hook_type="command", success=True, output="ok"),
    ])
    assert agg2.blocked is False
    assert agg2.all_passed is True

    agg3 = AggregatedHookResult(results=[])
    assert agg3.blocked is False
    assert agg3.all_passed is True
    assert agg3.reason == ""

    print("9. AggregatedHookResult: OK")


# ===================================================================
# Test 9: summary()
# ===================================================================
def test_summary():
    config = {
        "pre_tool_use": [
            {"type": "command", "command": "echo x", "matcher": "bash"},
        ],
    }
    reg = load_hook_registry(config)
    summary = reg.summary()
    assert "pre_tool_use" in summary
    assert "matcher=bash" in summary
    print("10. Summary: OK")


# ===================================================================
# Test 10: HookResult creation
# ===================================================================
def test_hook_result():
    hr = HookResult(
        hook_type="command",
        success=True,
        output="hello world",
        metadata={"returncode": 0},
    )
    assert hr.hook_type == "command"
    assert hr.success is True
    assert hr.blocked is False
    assert hr.metadata["returncode"] == 0

    hr2 = HookResult(
        hook_type="prompt",
        success=False,
        output="bad",
        blocked=True,
        reason="model says no",
    )
    assert hr2.blocked is True
    assert hr2.reason == "model says no"

    print("11. HookResult: OK")


# ===================================================================
# Test 12: TOOL_FAILED event exists
# ===================================================================
def test_tool_failed_event():
    from miniharness.hooks.events import HookEvent
    assert hasattr(HookEvent, "TOOL_FAILED")
    assert HookEvent.TOOL_FAILED.value == "tool_failed"
    print("12. TOOL_FAILED event: OK")


# ===================================================================
# Test 13: ConfirmHookDefinition
# ===================================================================
def test_confirm_hook_schema():
    from miniharness.hooks.schemas import ConfirmHookDefinition

    ch = ConfirmHookDefinition(
        message="Approve $TOOL_NAME?",
        matcher="bash:*rm*",
        timeout_seconds=60,
    )
    assert ch.type == "confirm"
    assert ch.message == "Approve $TOOL_NAME?"
    assert ch.matcher == "bash:*rm*"
    assert ch.block_on_failure is True  # default for confirm
    print("13. ConfirmHookDefinition: OK")


# ===================================================================
# Test 14: Presets — dangerous commands
# ===================================================================
def test_dangerous_command_preset():
    from miniharness.hooks.presets import (
        dangerous_command_preset,
        DANGEROUS_COMMAND_PATTERNS,
    )
    config = dangerous_command_preset()
    assert "pre_tool_use" in config
    hooks = config["pre_tool_use"]
    assert len(hooks) == len(DANGEROUS_COMMAND_PATTERNS)
    # All should be command type with block_on_failure=True.
    for h in hooks:
        assert h["type"] == "command"
        assert h["block_on_failure"] is True
        assert "matcher" in h
    print(f"14. Dangerous command preset: OK ({len(hooks)} patterns)")


# ===================================================================
# Test 15: Presets — sensitive files
# ===================================================================
def test_sensitive_file_preset():
    from miniharness.hooks.presets import (
        sensitive_file_preset,
        SENSITIVE_FILE_PATTERNS,
    )
    config = sensitive_file_preset()
    hooks = config["pre_tool_use"]
    assert len(hooks) == len(SENSITIVE_FILE_PATTERNS)
    # All should block on_failure.
    for h in hooks:
        assert h["block_on_failure"] is True
        assert "write_file" in h["matcher"] or "edit_file" in h["matcher"]
    print(f"15. Sensitive file preset: OK ({len(hooks)} patterns)")


# ===================================================================
# Test 16: Presets — approval
# ===================================================================
def test_approval_preset():
    from miniharness.hooks.presets import approval_preset, APPROVAL_PATTERNS
    config = approval_preset()
    hooks = config["pre_tool_use"]
    assert len(hooks) == len(APPROVAL_PATTERNS)
    for h in hooks:
        assert h["type"] == "confirm"
    print(f"16. Approval preset: OK ({len(hooks)} patterns)")


# ===================================================================
# Test 17: Presets — production bundle
# ===================================================================
def test_production_presets():
    from miniharness.hooks.presets import production_presets
    config = production_presets(enable_code_review=False)
    assert "pre_tool_use" in config
    assert "post_tool_use" in config
    assert "tool_failed" in config
    assert "session_start" in config
    # Should have audit logging hooks.
    assert len(config["post_tool_use"]) >= 1
    assert len(config["tool_failed"]) >= 1
    print(f"17. Production presets: OK ({sum(len(v) for v in config.values())} total hooks)")


# ===================================================================
# Test 18: Presets load into registry
# ===================================================================
def test_presets_loadable():
    from miniharness.hooks.presets import production_presets
    config = production_presets(enable_code_review=False)
    reg = load_hook_registry(config)
    # Should have hooks for all major events.
    assert reg.total_count > 0
    assert len(reg.get(HookEvent.PRE_TOOL_USE)) > 0
    assert len(reg.get(HookEvent.POST_TOOL_USE)) > 0
    assert len(reg.get(HookEvent.TOOL_FAILED)) > 0
    assert len(reg.get(HookEvent.SESSION_START)) > 0
    print(f"18. Presets → registry: OK ({reg.total_count} hooks loaded)")


# ===================================================================
# Test 19: AuditLogger
# ===================================================================
def test_audit_logger():
    import tempfile, os
    from miniharness.hooks.audit import AuditLogger

    tmpdir = tempfile.mkdtemp()
    try:
        logger = AuditLogger(tmpdir)
        logger.log("pre_tool_use", tool_name="bash", session_id="test123")
        logger.log("tool_failed", tool_name="bash", error="timeout")

        entries = logger.tail(10)
        assert len(entries) == 2
        assert entries[0]["tool_name"] == "bash"
        assert entries[1]["event"] == "tool_failed"

        # Search.
        failures = logger.search(event="tool_failed")
        assert len(failures) == 1
        assert failures[0]["error"] == "timeout"
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    print("19. AuditLogger: OK")


# ===================================================================
# Test 20: Confirm hook registry + matcher integration
# ===================================================================
def test_confirm_in_registry():
    config = {
        "pre_tool_use": [
            {"type": "confirm", "message": "Approve $TOOL_NAME?",
             "matcher": "bash:*rm*"},
        ],
    }
    reg = load_hook_registry(config)
    hooks = reg.get(HookEvent.PRE_TOOL_USE)
    assert len(hooks) == 1
    from miniharness.hooks.schemas import ConfirmHookDefinition
    assert isinstance(hooks[0], ConfirmHookDefinition)
    assert hooks[0].message == "Approve $TOOL_NAME?"
    print("20. Confirm in registry: OK")


# ===================================================================
# Run all tests
# ===================================================================
if __name__ == "__main__":
    test_registry()
    test_matcher_basic()
    test_matcher_tool_input()
    test_matcher_cross_tool()
    test_extract_subjects()
    test_matcher_glob()
    test_parser()
    test_injection()
    asyncio.run(test_command_hook())
    asyncio.run(test_blocking())
    asyncio.run(test_runtime_matcher())
    test_aggregated()
    test_summary()
    test_hook_result()
    test_tool_failed_event()
    test_confirm_hook_schema()
    test_dangerous_command_preset()
    test_sensitive_file_preset()
    test_approval_preset()
    test_production_presets()
    test_presets_loadable()
    test_audit_logger()
    test_confirm_in_registry()
    print()
    print("=== ALL hook system integration tests passed! ===")
