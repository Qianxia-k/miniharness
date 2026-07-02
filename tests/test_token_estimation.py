import pytest

from miniharness.config import apply_cli_overrides, load_settings
from miniharness.context.budget import ContextBudget, count_tokens
from miniharness.context.compiler import ContextCompiler
from miniharness.config.settings import Settings
from miniharness.commands.builtin import cmd_tokens
from miniharness.commands.types import CommandContext
from miniharness.messages import Conversation, Message
from miniharness.ui.runtime import RuntimeController
from miniharness.services.token_estimation import estimate_tokens, tokenizer_name_for_model


def test_estimate_tokens_falls_back_without_tiktoken_requirement():
    assert estimate_tokens("hello world", model="unknown-model") >= 1


def test_count_tokens_includes_tool_calls_and_tool_results():
    messages = [
        {"role": "user", "content": "read the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "large tool output"},
    ]

    assert count_tokens(messages, model="gpt-4o-mini") > count_tokens(
        [{"role": "user", "content": "read the file"}],
        model="gpt-4o-mini",
    )


def test_budget_snapshot_reports_breakdown():
    budget = ContextBudget.for_model("gpt-4o-mini", ratio=0.8)
    messages = [{"role": "user", "content": "hello"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]

    snapshot = budget.snapshot(messages, tools=tools)

    assert snapshot["token_count"] == snapshot["total_used"]
    assert snapshot["message_tokens"] > 0
    assert snapshot["tool_tokens"] > 0
    assert snapshot["response_reserve_tokens"] == budget.response_reserve_tokens
    assert snapshot["context_window"] == budget.total
    assert snapshot["soft_limit"] == budget.max_tokens
    assert snapshot["tokenizer"] == tokenizer_name_for_model("gpt-4o-mini")


def test_context_budget_ratio_is_used_by_budget_factory():
    settings = Settings()
    budget = ContextBudget.for_model("gpt-4o-mini", ratio=settings.context_budget_ratio)

    assert budget.max_tokens == int(budget.total * settings.context_budget_ratio)


def test_default_context_budget_ratio_is_production_safe():
    assert Settings().context_budget_ratio == 0.8


def test_context_budget_ratio_can_be_set_from_env(monkeypatch):
    monkeypatch.setenv("MINIHARNESS_CONTEXT_BUDGET_RATIO", "0.05")

    assert load_settings().context_budget_ratio == 0.05


def test_context_budget_ratio_cli_override_wins():
    settings = apply_cli_overrides(Settings(), context_budget_ratio=0.05)

    assert settings.context_budget_ratio == 0.05


def test_tokens_command_includes_last_compile_trace(tmp_path):
    runtime = RuntimeController(cwd=tmp_path, settings=Settings())
    runtime.loop.last_context_stats = {
        "context_trace": {
            "message_count": 3,
            "tool_count": 12,
            "system_prompt_chars": 1234,
            "attachment_count": 2,
            "attachment_types": ["work_log", "background_tasks"],
            "compacted": False,
        }
    }
    ctx = CommandContext(
        loop=runtime.loop,
        console=None,
        cwd=tmp_path,
        skill_registry=runtime.loop.skill_registry,
        hook_registry=runtime.loop.hook_registry,
        tool_registry=runtime.loop.tools,
    )

    result = cmd_tokens("", ctx)

    assert result.message is not None
    assert "last compile:" in result.message
    assert "tools exposed: 12" in result.message
    assert "attachment types: work_log, background_tasks" in result.message


async def _never_called_llm(**kwargs):
    raise AssertionError("context collapse should not need an LLM call")


@pytest.mark.asyncio
async def test_compiler_marks_real_compaction_when_budget_is_exceeded():
    budget = ContextBudget(
        model="gpt-4o-mini",
        total=4096,
        max_tokens=1200,
        ratio=0.8,
        response_reserve_tokens=0,
    )
    compiler = ContextCompiler(budget=budget, llm_stream=_never_called_llm)
    conversation = Conversation()
    conversation.append(Message(role="system", content="system"))
    conversation.append(Message(role="user", content="x" * 8000))

    packet = await compiler.compile(conversation, tools=[])

    assert packet.stats["compacted"] is True
    assert packet.stats["tier2_context_collapse"] is True
    assert "[collapsed" in packet.messages[1]["content"]
    assert packet.stats["context_trace"]["compacted"] is True
    assert packet.stats["context_trace"]["message_count"] == len(packet.messages)


@pytest.mark.asyncio
async def test_compiler_records_context_trace_without_compaction():
    budget = ContextBudget(
        model="gpt-4o-mini",
        total=4096,
        max_tokens=3000,
        ratio=0.8,
        response_reserve_tokens=0,
    )
    compiler = ContextCompiler(budget=budget, llm_stream=_never_called_llm)
    conversation = Conversation()
    conversation.append(Message(role="system", content="system prompt"))
    conversation.append(Message(role="user", content="hello"))
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    attachments = [{"type": "work_log", "content": "read README"}]

    packet = await compiler.compile(conversation, tools=tools, attachments=attachments)
    trace = packet.stats["context_trace"]

    assert trace["message_count"] == 2
    assert trace["system_prompt_chars"] == len("system prompt")
    assert trace["tool_count"] == 1
    assert trace["attachment_count"] == 1
    assert trace["attachment_types"] == ["work_log"]
    assert trace["compacted"] is False


@pytest.mark.asyncio
async def test_compiler_emits_compaction_progress_events():
    events: list[dict] = []

    async def collect(event: dict) -> None:
        events.append(event)

    budget = ContextBudget(
        model="gpt-4o-mini",
        total=4096,
        max_tokens=1200,
        ratio=0.8,
        response_reserve_tokens=0,
    )
    compiler = ContextCompiler(
        budget=budget,
        llm_stream=_never_called_llm,
        compact_progress=collect,
    )
    conversation = Conversation()
    conversation.append(Message(role="system", content="system"))
    conversation.append(Message(role="user", content="x" * 8000))

    await compiler.compile(conversation, tools=[])

    assert events[0]["phase"] == "start"
    assert any(event["phase"] == "tier_start" for event in events)
    assert any(event["phase"] == "tier_end" for event in events)
    assert events[-1]["phase"] == "end"
    assert events[-1]["compacted"] is True
