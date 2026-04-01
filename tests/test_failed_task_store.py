"""Tests for FailedTaskStore and orphan recovery."""

import asyncio
import json
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from mentat.core.queue import (
    BackgroundProcessor,
    FailedTaskStore,
    MAX_RETRIES,
    ProcessingTask,
)
from mentat.core.watcher import MentatWatcher
from mentat.probes.base import Chunk, ProbeResult, StructureInfo, TopicInfo


# ── FailedTaskStore unit tests ───────────────────────────────────────


class TestFailedTaskStore:
    def test_empty_store(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        assert store.all_failed() == {}
        assert store.get("nonexistent") is None
        assert store.get_retry_count("nonexistent") == 0
        assert not store.is_exhausted("nonexistent")

    def test_record_failure(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_failure("doc-1", error="API timeout", source="watcher:chat_history", filename="session.jsonl@0")

        entry = store.get("doc-1")
        assert entry is not None
        assert entry["retry_count"] == 1
        assert entry["last_error"] == "API timeout"
        assert entry["source"] == "watcher:chat_history"
        assert entry["filename"] == "session.jsonl@0"

    def test_record_multiple_failures_increments_count(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_failure("doc-1", error="err1")
        store.record_failure("doc-1", error="err2")
        store.record_failure("doc-1", error="err3")

        entry = store.get("doc-1")
        assert entry["retry_count"] == 3
        assert entry["last_error"] == "err3"

    def test_is_exhausted(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        for i in range(MAX_RETRIES):
            assert not store.is_exhausted("doc-1")
            store.record_failure("doc-1", error=f"err{i}")
        assert store.is_exhausted("doc-1")

    def test_record_success_removes_entry(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_failure("doc-1", error="err")
        assert store.get("doc-1") is not None

        store.record_success("doc-1")
        assert store.get("doc-1") is None

    def test_record_success_noop_for_unknown(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_success("nonexistent")  # Should not raise

    def test_persistence_across_instances(self, tmp_path):
        store1 = FailedTaskStore(str(tmp_path))
        store1.record_failure("doc-1", error="err", source="src")

        store2 = FailedTaskStore(str(tmp_path))
        entry = store2.get("doc-1")
        assert entry is not None
        assert entry["retry_count"] == 1
        assert entry["source"] == "src"

    def test_corrupted_file_starts_fresh(self, tmp_path):
        (tmp_path / "failed_tasks.json").write_text("not valid json")
        store = FailedTaskStore(str(tmp_path))
        assert store.all_failed() == {}

    def test_preserves_first_failed_at(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_failure("doc-1", error="first")
        first_ts = store.get("doc-1")["first_failed_at"]

        time.sleep(0.01)
        store.record_failure("doc-1", error="second")
        assert store.get("doc-1")["first_failed_at"] == first_ts

    def test_remove(self, tmp_path):
        store = FailedTaskStore(str(tmp_path))
        store.record_failure("doc-1", error="err")
        store.remove("doc-1")
        assert store.get("doc-1") is None


# ── Orphan recovery tests ────────────────────────────────────────────


def _make_probe_result():
    return ProbeResult(
        file_type="markdown",
        filename="test.md",
        topic=TopicInfo(title="Test"),
        structure=StructureInfo(toc=[]),
        chunks=[Chunk(content="chunk content", index=0, section="A")],
        stats={"is_full_content": False},
    )


def _mock_mentat_for_recovery(tmp_path, stubs=None, has_chunks_set=None):
    """Create a mock Mentat that returns specified stubs and chunk status."""
    m = MagicMock()
    m.config.db_path = str(tmp_path / "db")
    Path(m.config.db_path).mkdir(parents=True, exist_ok=True)
    m.embeddings = MagicMock()
    m.embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
    m.librarian = MagicMock()
    m.librarian.summarize_chunks = AsyncMock(return_value=["summary"])
    m.storage = MagicMock()
    m.storage._ensure_chunks_table = MagicMock()
    m.storage.add_chunks = MagicMock()
    m.storage.list_docs.return_value = stubs or []

    has_chunks_set = has_chunks_set or set()
    m.storage.has_chunks.side_effect = lambda doc_id: doc_id in has_chunks_set

    def _get_stub(doc_id):
        for s in (stubs or []):
            if s.get("id") == doc_id:
                return s
        return None
    m.storage.get_stub.side_effect = _get_stub

    return m


class TestOrphanRecovery:
    @pytest.mark.asyncio
    async def test_no_stubs_no_recovery(self, tmp_path):
        m = _mock_mentat_for_recovery(tmp_path, stubs=[])
        processor = BackgroundProcessor(m, max_concurrent=1)
        await processor._recover_orphan_stubs()
        # No tasks should be queued
        assert processor.queue._tasks == {}

    @pytest.mark.asyncio
    async def test_stub_with_chunks_not_recovered(self, tmp_path):
        probe = _make_probe_result()
        stubs = [{"id": "doc-1", "filename": "test.md", "probe_json": probe.model_dump_json()}]
        m = _mock_mentat_for_recovery(tmp_path, stubs=stubs, has_chunks_set={"doc-1"})

        processor = BackgroundProcessor(m, max_concurrent=1)
        await processor._recover_orphan_stubs()
        assert processor.queue._tasks == {}

    @pytest.mark.asyncio
    async def test_orphan_stub_requeued(self, tmp_path):
        probe = _make_probe_result()
        stubs = [{
            "id": "doc-orphan",
            "filename": "test.md",
            "probe_json": probe.model_dump_json(),
            "source": "watcher:chat_history",
            "metadata_json": "{}",
        }]
        m = _mock_mentat_for_recovery(tmp_path, stubs=stubs, has_chunks_set=set())

        processor = BackgroundProcessor(m, max_concurrent=1)
        await processor._recover_orphan_stubs()

        assert "doc-orphan" in processor.queue._tasks
        task = processor.queue._tasks["doc-orphan"]
        assert task.status == "pending"
        assert task.priority == -1  # Lower than normal tasks

    @pytest.mark.asyncio
    async def test_exhausted_task_not_requeued(self, tmp_path):
        probe = _make_probe_result()
        stubs = [{"id": "doc-ex", "filename": "test.md", "probe_json": probe.model_dump_json()}]
        m = _mock_mentat_for_recovery(tmp_path, stubs=stubs, has_chunks_set=set())

        processor = BackgroundProcessor(m, max_concurrent=1)
        # Exhaust retries
        for i in range(MAX_RETRIES):
            processor.failed_store.record_failure("doc-ex", error=f"err{i}")

        await processor._recover_orphan_stubs()
        assert "doc-ex" not in processor.queue._tasks

    @pytest.mark.asyncio
    async def test_success_clears_failed_store(self, tmp_path):
        probe = _make_probe_result()
        stubs = [{"id": "doc-ok", "filename": "test.md", "probe_json": probe.model_dump_json()}]
        m = _mock_mentat_for_recovery(tmp_path, stubs=stubs, has_chunks_set={"doc-ok"})

        processor = BackgroundProcessor(m, max_concurrent=1)
        processor.failed_store.record_failure("doc-ok", error="old error")

        await processor._recover_orphan_stubs()
        # Success should clear the failed entry
        assert processor.failed_store.get("doc-ok") is None

    @pytest.mark.asyncio
    async def test_process_task_failure_records_in_store(self, tmp_path):
        m = _mock_mentat_for_recovery(tmp_path)
        m.embeddings.embed_batch = AsyncMock(side_effect=Exception("API Error"))

        processor = BackgroundProcessor(m, max_concurrent=1)
        task = ProcessingTask(
            doc_id="doc-fail",
            probe_result=_make_probe_result(),
            chunk_extra_fields={"source": "watcher:chat_history"},
        )
        await processor._process_task(task)

        assert task.status == "failed"
        entry = processor.failed_store.get("doc-fail")
        assert entry is not None
        assert "API Error" in entry["last_error"]
        assert entry["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_process_task_success_clears_store(self, tmp_path):
        m = _mock_mentat_for_recovery(tmp_path)
        processor = BackgroundProcessor(m, max_concurrent=1)
        processor.failed_store.record_failure("doc-ok", error="old")

        task = ProcessingTask(
            doc_id="doc-ok",
            probe_result=_make_probe_result(),
        )
        await processor._process_task(task)

        assert task.status == "completed"
        assert processor.failed_store.get("doc-ok") is None


# ── Watcher offset verification tests ────────────────────────────────


def _mock_mentat_for_watcher(tmp_path, collection_docs=None, stubs_with_chunks=None):
    """Mock Mentat for watcher offset verification tests."""
    m = MagicMock()
    m.config.storage_dir = str(tmp_path / "storage")
    m.config.db_path = str(tmp_path / "db")
    Path(m.config.storage_dir).mkdir(parents=True, exist_ok=True)
    Path(m.config.db_path).mkdir(parents=True, exist_ok=True)
    m.add = AsyncMock(return_value="doc-new")

    collection_docs = collection_docs or {}
    stubs_with_chunks = stubs_with_chunks or set()

    def _get_collection(name):
        if name in collection_docs:
            return {"doc_ids": collection_docs[name]}
        return None
    m.collections_store.get.side_effect = _get_collection

    def _get_stub(doc_id):
        # Return a stub with filename matching the doc_id pattern
        return {"filename": doc_id}  # Simplified
    m.storage.get_stub.side_effect = _get_stub

    m.storage.has_chunks.side_effect = lambda doc_id: doc_id in stubs_with_chunks

    return m


def _write_jsonl(path, lines):
    with open(path, "a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


class TestOffsetVerification:
    def test_empty_collection_resets_offset(self, tmp_path):
        """If collection has no docs, offset is stale — reset to 0."""
        m = _mock_mentat_for_watcher(tmp_path, collection_docs={"chat_history": []})

        watcher = MentatWatcher(m)
        watcher._configs = {
            "chat_history": {"watch_mode": "append", "watch_paths": [], "watch_ignore": []}
        }
        fake_path = "/some/sessions/session.jsonl"
        watcher._offsets[fake_path] = 5000

        watcher._verify_offset_integrity("chat_history", fake_path)
        assert watcher._offsets[fake_path] == 0

    def test_docs_with_chunks_keeps_offset(self, tmp_path):
        """If matching doc has chunks, offset is valid — keep it."""
        m = _mock_mentat_for_watcher(
            tmp_path,
            collection_docs={"chat_history": ["session.jsonl@0"]},
            stubs_with_chunks={"session.jsonl@0"},
        )

        watcher = MentatWatcher(m)
        watcher._configs = {
            "chat_history": {"watch_mode": "append", "watch_paths": [], "watch_ignore": []}
        }
        fake_path = "/some/sessions/session.jsonl"
        watcher._offsets[fake_path] = 5000

        watcher._verify_offset_integrity("chat_history", fake_path)
        assert watcher._offsets[fake_path] == 5000

    def test_docs_without_chunks_resets_offset(self, tmp_path):
        """If matching doc exists but has no chunks (failed embed), reset offset."""
        m = _mock_mentat_for_watcher(
            tmp_path,
            collection_docs={"chat_history": ["session.jsonl@0"]},
            stubs_with_chunks=set(),  # No chunks
        )

        watcher = MentatWatcher(m)
        watcher._configs = {
            "chat_history": {"watch_mode": "append", "watch_paths": [], "watch_ignore": []}
        }
        fake_path = "/some/sessions/session.jsonl"
        watcher._offsets[fake_path] = 5000

        watcher._verify_offset_integrity("chat_history", fake_path)
        assert watcher._offsets[fake_path] == 0

    def test_unknown_collection_noop(self, tmp_path):
        """If collection doesn't exist, do nothing."""
        m = _mock_mentat_for_watcher(tmp_path, collection_docs={})

        watcher = MentatWatcher(m)
        watcher._configs = {
            "chat_history": {"watch_mode": "append", "watch_paths": [], "watch_ignore": []}
        }
        fake_path = "/some/sessions/session.jsonl"
        watcher._offsets[fake_path] = 5000

        watcher._verify_offset_integrity("chat_history", fake_path)
        assert watcher._offsets[fake_path] == 5000  # Unchanged

    def test_zero_offset_stays_zero(self, tmp_path):
        """Offset of 0 shouldn't trigger a warning."""
        m = _mock_mentat_for_watcher(tmp_path, collection_docs={"chat_history": []})

        watcher = MentatWatcher(m)
        watcher._configs = {
            "chat_history": {"watch_mode": "append", "watch_paths": [], "watch_ignore": []}
        }
        fake_path = "/some/sessions/session.jsonl"
        watcher._offsets[fake_path] = 0

        watcher._verify_offset_integrity("chat_history", fake_path)
        assert watcher._offsets[fake_path] == 0
