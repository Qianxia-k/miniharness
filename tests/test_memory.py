"""Integration tests for the production-grade memory system."""
import json, tempfile, time
from pathlib import Path
from unittest.mock import patch

from miniharness.memory.base import MemoryStore, _tokenise
from miniharness.memory.semantic import SemanticStore
from miniharness.memory.episodic import EpisodicStore
from miniharness.memory.store import get_memory_dir


# ===================================================================
# Test 1: _tokenise
# ===================================================================
def test_tokenise():
    assert _tokenise("FastAPI JWT auth") == ["fastapi", "jwt", "auth"]
    assert _tokenise("a b c") == []  # single chars filtered out
    assert _tokenise("") == []
    print("1. Tokeniser: OK")


# ===================================================================
# Test 2: MemoryStore atomic write — no corruption
# ===================================================================
def test_atomic_write(tmp_path):
    """Verify atomic writes: temp file → rename, no partial writes."""
    store_path = tmp_path / "test.json"

    class TestStore(MemoryStore):
        _filename = "test.json"

        def _entry_search_text(self, entry):
            return entry.get("text", "")

    # Patch get_memory_dir to return tmp_path.
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = TestStore("/fake/project")

        # Write some entries.
        entries = [{"id": "1", "text": "hello", "timestamp": time.time()}]
        store._write_all(entries)

        # Verify the file exists and is valid JSON.
        assert store_path.exists()
        data = json.loads(store_path.read_text())
        assert len(data) == 1
        assert data[0]["text"] == "hello"

        # Verify no temp files left behind.
        temps = list(tmp_path.glob(".*.json.*"))
        assert len(temps) == 0, f"Temp files left behind: {temps}"

    print("2. Atomic write: OK")


# ===================================================================
# Test 3: SemanticStore add + search + backward compat
# ===================================================================
def test_semantic_store(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project", max_entries=100)

        # Add facts.
        id1 = store.add("Project uses FastAPI", tags=["tech-stack"])
        id2 = store.add("Auth module uses JWT tokens", tags=["auth", "security"])
        id3 = store.add("Database is PostgreSQL 15", tags=["tech-stack", "db"])

        assert len(id1) == 12
        assert len(id2) == 12

        # Search.
        results = store.search("JWT", limit=5)
        assert len(results) == 1
        assert "JWT" in results[0]["fact"]

        results = store.search("FastAPI", limit=5)
        assert len(results) == 1
        assert results[0]["tags"] == ["tech-stack"]

        # Tag search should also work (tags are in _entry_search_text).
        results = store.search("auth", limit=5)
        assert len(results) >= 1
        assert any("auth" in r.get("tags", []) for r in results)

        # list_all.
        all_entries = store.list_all(limit=50)
        assert len(all_entries) == 3

        # count.
        assert store.count == 3
    print("3. SemanticStore: OK")


# ===================================================================
# Test 4: EpisodicStore log + search + backward compat
# ===================================================================
def test_episodic_store(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = EpisodicStore("/fake/project", max_entries=100, ttl_seconds=None)

        # Log episodes.
        id1 = store.log(
            task="Refactored auth module",
            summary="Extracted JWT logic into middleware.py",
            files_touched=["src/auth.py", "src/middleware.py"],
            outcome="success",
        )
        id2 = store.log(
            task="Added unit tests",
            summary="Wrote 25 tests for auth module",
            files_touched=["tests/test_auth.py"],
            outcome="success",
        )
        assert len(id1) == 12

        # Search.
        results = store.search("JWT middleware", limit=5)
        assert len(results) == 1
        assert results[0]["task"] == "Refactored auth module"

        # Search by file path (unique to second episode).
        results = store.search("25 tests", limit=5)
        assert len(results) == 1
        assert results[0]["task"] == "Added unit tests"

        # list_all.
        all_entries = store.list_all(limit=50)
        assert len(all_entries) == 2

        # count.
        assert store.count == 2
    print("4. EpisodicStore: OK")


# ===================================================================
# Test 5: max_entries — oldest evicted first (FIFO)
# ===================================================================
def test_max_entries_enforcement(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project", max_entries=3)

        # Add 5 entries.
        for i in range(5):
            store.add(f"Fact number {i}")
            time.sleep(0.01)  # ensure distinct timestamps

        # Should only retain the 3 newest.
        entries = store.list_all(limit=50)
        assert len(entries) == 3, f"Expected 3, got {len(entries)}"
        # Newest first → fact 4, 3, 2 (0 and 1 evicted).
        facts = [e["fact"] for e in entries]
        assert "Fact number 4" in facts
        assert "Fact number 3" in facts
        assert "Fact number 2" in facts
        assert "Fact number 0" not in facts
        assert "Fact number 1" not in facts
    print("5. max_entries enforcement: OK")


# ===================================================================
# Test 6: TTL expiry
# ===================================================================
def test_ttl_expiry(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        # Short TTL: 1 second.
        store = SemanticStore("/fake/project", max_entries=100, ttl_seconds=1)

        store.add("Recent fact")
        assert store.count == 1

        # Wait for expiry.
        time.sleep(1.5)

        # Adding a new entry triggers prune — old entry should be gone.
        store.add("New fact")
        entries = store.list_all(limit=50)
        assert len(entries) == 1
        assert entries[0]["fact"] == "New fact"
    print("6. TTL expiry: OK")


# ===================================================================
# Test 7: EpisodicStore default TTL (90 days)
# ===================================================================
def test_episodic_default_ttl():
    store = EpisodicStore("/tmp/test", max_entries=10)
    # Default TTL should be positive (at least 1 day).
    assert store.ttl_seconds is not None
    assert store.ttl_seconds >= 86400
    print(f"7. EpisodicStore default TTL: OK ({store.ttl_seconds}s = {store.ttl_seconds/86400:.0f}d)")


# ===================================================================
# Test 8: SemanticStore default values
# ===================================================================
def test_semantic_defaults():
    store = SemanticStore("/tmp/test")
    assert store.max_entries == 500
    assert store.ttl_seconds is None  # no TTL by default for facts
    print("8. SemanticStore defaults: OK")


# ===================================================================
# Test 9: Prune handles empty list
# ===================================================================
def test_prune_empty(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project", max_entries=10, ttl_seconds=3600)
        result = store._prune([])
        assert result == []
    print("9. Prune empty list: OK")


# ===================================================================
# Test 10: Corrupt JSON file → returns empty
# ===================================================================
def test_corrupt_json(tmp_path):
    store_path = tmp_path / "semantic.json"
    store_path.write_text("this is not json {{{")

    class TestStore(MemoryStore):
        _filename = "semantic.json"
        def _entry_search_text(self, entry):
            return entry.get("text", "")

    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = TestStore("/fake/project")
        entries = store._read_all()
        assert entries == []  # corrupt file → empty list, no crash
    print("10. Corrupt JSON recovery: OK")


def test_semantic_duplicate_refreshes_existing_entry(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project", max_entries=100)

        id1 = store.add("Project uses FastAPI", tags=["tech-stack"])
        time.sleep(0.01)
        id2 = store.add("Project uses FastAPI", tags=["python"])

        assert id1 == id2
        entries = store.list_all(limit=50)
        assert len(entries) == 1
        assert entries[0]["id"] == id1
        assert entries[0]["status"] == "active"
        assert entries[0]["tags"] == ["tech-stack", "python"]
        assert entries[0]["updated_at"] >= entries[0]["created_at"]
    print("11. Semantic duplicate refresh: OK")


def test_semantic_supersedes_disables_old_entry(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = SemanticStore("/fake/project", max_entries=100)

        old_id = store.add("Auth module uses JWT with HS256", tags=["auth"])
        new_id = store.add(
            "Auth module uses JWT with RS256",
            tags=["auth"],
            supersedes=[old_id],
        )

        active = store.list_all(limit=50)
        assert [entry["id"] for entry in active] == [new_id]
        assert active[0]["supersedes"] == [old_id]

        all_entries = store.list_all(limit=50, include_disabled=True)
        old = next(entry for entry in all_entries if entry["id"] == old_id)
        assert old["disabled"] is True
        assert old["status"] == "superseded"
        assert old["superseded_by"] == new_id
        assert store.count == 1
    print("12. Semantic supersede lifecycle: OK")


def test_episodic_duplicate_refreshes_existing_entry(tmp_path):
    with patch("miniharness.memory.base.get_memory_dir", return_value=tmp_path):
        store = EpisodicStore("/fake/project", max_entries=100, ttl_seconds=None)

        id1 = store.log(
            task="Refactored auth module",
            summary="Extracted JWT logic into middleware.py",
            files_touched=["src/auth.py"],
            outcome="partial",
        )
        time.sleep(0.01)
        id2 = store.log(
            task="Refactored auth module",
            summary="Extracted JWT logic into middleware.py",
            files_touched=["src/auth.py"],
            outcome="success",
        )

        assert id1 == id2
        entries = store.list_all(limit=50)
        assert len(entries) == 1
        assert entries[0]["outcome"] == "success"
        assert entries[0]["source"] == "manual"
    print("13. Episodic duplicate refresh: OK")


# ===================================================================
if __name__ == "__main__":
    import tempfile
    tmp = tempfile.mkdtemp()
    tmp_path = Path(tmp)
    try:
        test_tokenise()
        test_atomic_write(tmp_path)
        test_semantic_store(tmp_path)
        test_episodic_store(tmp_path)
        test_max_entries_enforcement(tmp_path)
        test_ttl_expiry(tmp_path)
        test_episodic_default_ttl()
        test_semantic_defaults()
        test_prune_empty(tmp_path)
        test_corrupt_json(tmp_path)
        test_semantic_duplicate_refreshes_existing_entry(tmp_path)
        test_semantic_supersedes_disables_old_entry(tmp_path)
        test_episodic_duplicate_refreshes_existing_entry(tmp_path)
        print()
        print("=== ALL memory system integration tests passed! ===")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
