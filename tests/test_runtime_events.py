import pytest

from miniharness.config.settings import Settings
from miniharness.llm import StreamComplete, TextDelta
from miniharness.messages import Message
from miniharness.runtime import AssistantDeltaEvent, RuntimeEventBus, StatusRuntimeEvent
from miniharness.ui.backend_host import _runtime_to_protocol_event
from miniharness.ui.runtime import RuntimeController


@pytest.mark.asyncio
async def test_runtime_event_bus_fans_out_to_sync_and_async_handlers():
    bus = RuntimeEventBus()
    seen: list[str] = []

    def sync_handler(event):
        seen.append(f"sync:{event.type}")

    async def async_handler(event):
        seen.append(f"async:{event.type}")

    unsubscribe = bus.subscribe(sync_handler)
    bus.subscribe(async_handler)

    await bus.emit(StatusRuntimeEvent(message="working"))

    assert seen == ["sync:status", "async:status"]

    unsubscribe()
    await bus.emit(StatusRuntimeEvent(message="again"))

    assert seen == ["sync:status", "async:status", "async:status"]


def test_runtime_status_event_maps_to_protocol_event():
    protocol_event = _runtime_to_protocol_event(StatusRuntimeEvent(message="ready"))

    assert protocol_event.type == "status"
    assert protocol_event.message == "ready"


@pytest.mark.asyncio
async def test_agent_loop_emits_assistant_delta_from_engine(tmp_path):
    bus = RuntimeEventBus()
    seen: list[object] = []
    bus.subscribe(seen.append)
    runtime = RuntimeController(cwd=tmp_path, settings=Settings(), event_bus=bus)

    async def fake_stream(*args, **kwargs):
        yield TextDelta("hello")
        yield StreamComplete(Message(role="assistant", content="hello"))

    runtime.loop._stream_fn = fake_stream  # type: ignore[attr-defined]

    try:
        message = await runtime.loop._call_llm(messages=[], tools=[], max_tokens_override=None)
    finally:
        await runtime.close()

    assert message is not None
    assert message.content == "hello"
    assert any(isinstance(event, AssistantDeltaEvent) and event.text == "hello" for event in seen)
