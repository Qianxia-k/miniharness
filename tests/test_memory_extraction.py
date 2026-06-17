"""Tests for automatic memory extraction."""
from unittest.mock import patch

import pytest

from miniharness.llm import StreamComplete
from miniharness.messages import Message
from miniharness.memory.semantic import SemanticStore
from miniharness.services.memory_extractor import _extract_facts, _parse_json_response


def test_parse_direct_json():
    r = _parse_json_response('{"facts": [{"fact": "x", "tags": ["a"]}]}')
    assert r is not None
    assert r["facts"][0]["fact"] == "x"


def test_parse_code_fence():
    r = _parse_json_response('```json\n{"facts": []}\n```')
    assert r is not None
    assert r["facts"] == []


def test_parse_empty():
    assert _parse_json_response("") is None


def test_parse_garbage():
    assert _parse_json_response("blah blah") is None


def test_parse_episode():
    r = _parse_json_response(
        '{"task": "Refactored auth", "summary": "Extracted JWT",'
        '"files_touched": ["src/auth.py"], "outcome": "success"}'
    )
    assert r is not None
    assert r["task"] == "Refactored auth"
    assert r["outcome"] == "success"
    assert len(r["files_touched"]) == 1


def test_parse_fact_lifecycle_fields():
    r = _parse_json_response(
        '{"facts": [{"fact": "Auth uses RS256", "tags": ["auth"], '
        '"confidence": 0.8, "supersedes": ["abc123"]}]}'
    )
    assert r is not None
    assert r["facts"][0]["supersedes"] == ["abc123"]
    assert r["facts"][0]["confidence"] == 0.8


@pytest.mark.asyncio
async def test_extract_facts_supersedes_existing_memory(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project")
        old_id = store.add("Auth module uses JWT with HS256", tags=["auth"])

        async def fake_llm(*, messages, tools, max_tokens_override):
            assert tools == []
            assert old_id in messages[-1]["content"]
            yield StreamComplete(
                Message(
                    role="assistant",
                    content=(
                        '{"facts": [{"fact": "Auth module uses JWT with RS256", '
                        '"tags": ["auth"], "confidence": 0.9, '
                        f'"supersedes": ["{old_id}"]}}]}}'
                    ),
                )
            )

        result = await _extract_facts(
            [
                {"role": "user", "content": "Auth is RS256 now"},
                {"role": "assistant", "content": "Updated."},
            ],
            fake_llm,
            "/fake/project",
        )

        assert len(result) == 1
        active = store.list_all(limit=10)
        assert len(active) == 1
        assert active[0]["fact"] == "Auth module uses JWT with RS256"
        assert active[0]["supersedes"] == [old_id]

        all_entries = store.list_all(limit=10, include_disabled=True)
        old = next(entry for entry in all_entries if entry["id"] == old_id)
        assert old["status"] == "superseded"
        assert old["disabled"] is True
