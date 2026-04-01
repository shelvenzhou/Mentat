"""E2E test for chat history search via Mentat.

Requires a running Mentat server (``uv run python -m mentat.server``).
Skip with: ``pytest -m "not e2e"``

Tests the full pipeline:
  1. Create chat_history collection with append mode
  2. Write mock session JSONL
  3. Wait for watcher to index
  4. Search with metadata_filter (session scope)
  5. Search globally
"""

import asyncio
import json
import os
import time
import pytest
import httpx
from pathlib import Path

pytestmark = pytest.mark.e2e

MENTAT_URL = os.environ.get("MENTAT_URL", "http://127.0.0.1:7832")
TIMEOUT = 60  # seconds to wait for indexing


def _is_server_up() -> bool:
    try:
        r = httpx.get(f"{MENTAT_URL}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture(scope="module")
def mentat_server():
    """Skip entire module if Mentat server is not running."""
    if not _is_server_up():
        pytest.skip("Mentat server not running")


@pytest.fixture(scope="module")
def sessions_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sessions")
    return d


@pytest.fixture(scope="module")
def setup_collection(mentat_server, sessions_dir):
    """Create the chat_history collection with append mode watching sessions_dir."""
    r = httpx.post(f"{MENTAT_URL}/collections", json={
        "name": "chat_history_test",
        "metadata": {
            "type": "test",
            "watch_mode": "append",
            "initial_scan_recent_days": 30,
            "watch_probe_config": {
                "filters": [
                    {"field": "type", "op": "eq", "value": "message"},
                    {"field": "message.role", "op": "in", "value": ["user", "assistant"]},
                ],
                "text_fields": ["message.content.text"],
                "label_field": "message.role",
                "group_size": 2,
            },
        },
        "watch_paths": [str(sessions_dir)],
        "watch_ignore": ["*.reset.*", "*.deleted.*"],
    }, timeout=10)
    assert r.status_code == 200, f"Failed to create collection: {r.text}"

    # Trigger watcher sync
    httpx.post(f"{MENTAT_URL}/watcher/sync", timeout=10)

    yield "chat_history_test"

    # Cleanup
    httpx.delete(f"{MENTAT_URL}/collections/chat_history_test", timeout=10)


def _write_session(sessions_dir: Path, session_id: str, messages: list[tuple[str, str]]):
    """Write a mock session JSONL file.

    messages: list of (role, text) tuples.
    """
    f = sessions_dir / f"{session_id}.jsonl"
    lines = [
        {"type": "session", "version": 3, "id": session_id, "timestamp": "2026-03-28T06:00:00Z"},
    ]
    for i, (role, text) in enumerate(messages):
        lines.append({
            "type": "message",
            "id": f"msg-{i}",
            "timestamp": f"2026-03-28T06:{i:02d}:00Z",
            "message": {
                "role": role,
                "content": [{"type": "text", "text": text}],
            },
        })
    with open(f, "w") as fp:
        for line in lines:
            fp.write(json.dumps(line) + "\n")
    return f


def _wait_for_index(collection: str, min_docs: int = 1, timeout: float = TIMEOUT):
    """Poll until the collection has at least min_docs documents."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = httpx.get(f"{MENTAT_URL}/collections/{collection}", timeout=5)
        if r.status_code == 200:
            data = r.json()
            if len(data.get("doc_ids", [])) >= min_docs:
                return True
        time.sleep(2)
    return False


class TestChatHistoryE2E:
    def test_session_indexed(self, setup_collection, sessions_dir):
        """Write a session file and verify it gets indexed."""
        _write_session(sessions_dir, "session-aaa", [
            ("user", "What is the capital of France?"),
            ("assistant", "The capital of France is Paris."),
        ])
        assert _wait_for_index(setup_collection, min_docs=1), \
            "Session not indexed within timeout"

    def test_search_global(self, setup_collection, sessions_dir):
        """Search across all sessions."""
        # Ensure indexed
        _wait_for_index(setup_collection, min_docs=1)

        r = httpx.post(f"{MENTAT_URL}/search", json={
            "query": "capital of France",
            "collection": setup_collection,
            "top_k": 5,
        }, timeout=30)
        assert r.status_code == 200
        results = r.json().get("results", [])
        assert len(results) > 0
        # Should contain the conversation about Paris
        texts = " ".join(r.get("content", "") for r in results)
        assert "Paris" in texts or "capital" in texts.lower()

    def test_search_with_session_filter(self, setup_collection, sessions_dir):
        """Write two sessions, search with metadata_filter for one."""
        _write_session(sessions_dir, "session-bbb", [
            ("user", "Tell me about quantum computing"),
            ("assistant", "Quantum computing uses qubits instead of classical bits."),
        ])
        _wait_for_index(setup_collection, min_docs=2)

        # Search scoped to session-bbb
        r = httpx.post(f"{MENTAT_URL}/search", json={
            "query": "quantum computing",
            "collection": setup_collection,
            "metadata_filter": {"session_id": "session-bbb"},
            "top_k": 5,
        }, timeout=30)
        assert r.status_code == 200
        results = r.json().get("results", [])
        assert len(results) > 0

        # Search scoped to session-aaa should NOT find quantum computing
        r2 = httpx.post(f"{MENTAT_URL}/search", json={
            "query": "quantum computing",
            "collection": setup_collection,
            "metadata_filter": {"session_id": "session-aaa"},
            "top_k": 5,
        }, timeout=30)
        results2 = r2.json().get("results", [])
        # Should either be empty or not contain quantum-related content
        if results2:
            texts = " ".join(r.get("content", "") for r in results2)
            assert "quantum" not in texts.lower()

    def test_incremental_append(self, setup_collection, sessions_dir):
        """Append to existing session → new content searchable."""
        session_file = sessions_dir / "session-aaa.jsonl"
        # Append new messages
        with open(session_file, "a") as f:
            for msg in [
                {"type": "message", "id": "msg-extra-1",
                 "timestamp": "2026-03-28T06:10:00Z",
                 "message": {"role": "user", "content": [{"type": "text", "text": "What about Berlin?"}]}},
                {"type": "message", "id": "msg-extra-2",
                 "timestamp": "2026-03-28T06:10:01Z",
                 "message": {"role": "assistant", "content": [{"type": "text", "text": "Berlin is the capital of Germany."}]}},
            ]:
                f.write(json.dumps(msg) + "\n")

        # Wait for new doc
        time.sleep(8)  # Wait for watcher cycle

        r = httpx.post(f"{MENTAT_URL}/search", json={
            "query": "Berlin Germany capital",
            "collection": setup_collection,
            "top_k": 5,
        }, timeout=30)
        assert r.status_code == 200
        results = r.json().get("results", [])
        texts = " ".join(r.get("content", "") for r in results)
        assert "Berlin" in texts or "Germany" in texts
