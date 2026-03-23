"""Integration tests for MentatWatcher lifecycle in Hub and Server."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from mentat.core.hub import Mentat, MentatConfig
from mentat.core.watcher import MentatWatcher
from mentat.server import create_app
from tests.conftest import FakeStorage, FakeEmbedding


# ── Hub lifecycle ──────────────────────────────────────────────────────


class TestHubWatcherLifecycle:
    """Watcher is created, started, and stopped with the Hub."""

    async def test_hub_has_watcher_attribute(self, tmp_path, monkeypatch):
        monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
        cfg = MentatConfig(
            db_path=str(tmp_path / "db"),
            storage_dir=str(tmp_path / "files"),
        )
        m = Mentat(cfg)
        assert isinstance(m.watcher, MentatWatcher)
        Mentat.reset()

    async def test_start_calls_watcher_start(self, tmp_path, monkeypatch):
        monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
        cfg = MentatConfig(
            db_path=str(tmp_path / "db"),
            storage_dir=str(tmp_path / "files"),
            max_concurrent_tasks=1,
        )
        m = Mentat(cfg)
        m.embeddings = FakeEmbedding()

        with patch.object(m.watcher, "start", new_callable=AsyncMock) as mock_start:
            await m.start()
            mock_start.assert_called_once()

        await m.shutdown()
        Mentat.reset()

    async def test_shutdown_calls_watcher_stop(self, tmp_path, monkeypatch):
        monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
        cfg = MentatConfig(
            db_path=str(tmp_path / "db"),
            storage_dir=str(tmp_path / "files"),
            max_concurrent_tasks=1,
        )
        m = Mentat(cfg)
        m.embeddings = FakeEmbedding()

        await m.start()

        with patch.object(m.watcher, "stop", new_callable=AsyncMock) as mock_stop:
            await m.shutdown()
            mock_stop.assert_called_once()

        Mentat.reset()


# ── Server collection CRUD triggers watcher sync ──────────────────────


@pytest.fixture
async def http_client(tmp_path, monkeypatch):
    monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
    cfg = MentatConfig(
        db_path=str(tmp_path / "db"),
        storage_dir=str(tmp_path / "files"),
        max_concurrent_tasks=2,
    )
    Mentat.reset()
    m = Mentat.get_instance(cfg)
    m.embeddings = FakeEmbedding()

    app = create_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await m.shutdown()
    Mentat.reset()


class TestServerWatcherSync:
    """Collection CRUD endpoints trigger watcher.sync()."""

    async def test_create_collection_syncs_watcher(self, http_client):
        m = Mentat.get_instance()
        with patch.object(m.watcher, "sync", new_callable=AsyncMock) as mock_sync:
            resp = await http_client.post(
                "/collections/watched",
                json={"watch_paths": ["/tmp/test-watch"]},
            )
            assert resp.status_code == 200
            mock_sync.assert_called_once()

    async def test_update_collection_syncs_watcher(self, http_client):
        await http_client.post("/collections/watched", json={})

        m = Mentat.get_instance()
        with patch.object(m.watcher, "sync", new_callable=AsyncMock) as mock_sync:
            resp = await http_client.put(
                "/collections/watched",
                json={"watch_paths": ["/tmp/updated-watch"]},
            )
            assert resp.status_code == 200
            mock_sync.assert_called_once()

    async def test_delete_collection_syncs_watcher(self, http_client):
        await http_client.post("/collections/watched", json={})

        m = Mentat.get_instance()
        with patch.object(m.watcher, "sync", new_callable=AsyncMock) as mock_sync:
            resp = await http_client.delete("/collections/watched")
            assert resp.status_code == 200
            mock_sync.assert_called_once()


# ── End-to-end: file change in watched dir gets indexed ───────────────

# watchfiles.awatch needs time to set up inotify watches before it can
# detect file changes.  This delay lets the OS-level watcher initialize.
_WATCHER_INIT_DELAY = 1.0
_POLL_INTERVAL = 0.25
_POLL_ATTEMPTS = 60


def _make_e2e_mentat(tmp_path, monkeypatch):
    """Helper: create a Mentat instance backed by fakes."""
    monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
    cfg = MentatConfig(
        db_path=str(tmp_path / "db"),
        storage_dir=str(tmp_path / "files"),
        max_concurrent_tasks=2,
    )
    Mentat.reset()
    m = Mentat(cfg)
    m.embeddings = FakeEmbedding()
    return m


class TestWatcherEndToEnd:
    """Watcher detects file changes and indexes them."""

    async def test_file_created_in_watched_dir_gets_indexed(self, tmp_path, monkeypatch):
        """Create a collection with watch_paths, write a file, verify indexing."""
        m = _make_e2e_mentat(tmp_path, monkeypatch)

        watch_dir = tmp_path / "memory"
        watch_dir.mkdir()

        m.collections_store.create("memory", watch_paths=[str(watch_dir)])

        original_add = m.add
        indexed_paths = []

        async def tracking_add(path, **kwargs):
            indexed_paths.append(path)
            return await original_add(path, **kwargs)

        m.add = tracking_add

        await m.start()

        try:
            # Let inotify watches initialize
            await asyncio.sleep(_WATCHER_INIT_DELAY)

            test_file = watch_dir / "note.md"
            test_file.write_text("# Important\nRemember this fact.")

            for _ in range(_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL)
                if any(str(test_file) in p for p in indexed_paths):
                    break

            assert any(str(test_file) in p for p in indexed_paths), (
                f"File {test_file} was not indexed. Indexed: {indexed_paths}"
            )

            # Verify doc is in the collection
            rec = m.collections_store.get("memory")
            assert rec is not None
            assert len(rec["doc_ids"]) > 0
        finally:
            await m.shutdown()
            Mentat.reset()

    async def test_file_modified_in_watched_dir_gets_reindexed(self, tmp_path, monkeypatch):
        """Modify a file in a watched dir — should be re-indexed."""
        m = _make_e2e_mentat(tmp_path, monkeypatch)

        watch_dir = tmp_path / "notes"
        watch_dir.mkdir()

        test_file = watch_dir / "doc.md"
        test_file.write_text("version 1")

        m.collections_store.create("notes", watch_paths=[str(watch_dir)])

        add_call_count = 0
        original_add = m.add

        async def counting_add(path, **kwargs):
            nonlocal add_call_count
            add_call_count += 1
            return await original_add(path, **kwargs)

        m.add = counting_add

        await m.start()

        try:
            await asyncio.sleep(_WATCHER_INIT_DELAY)

            test_file.write_text("version 2 — updated content")

            for _ in range(_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL)
                if add_call_count >= 1:
                    break

            assert add_call_count >= 1, "Modified file was not re-indexed"
        finally:
            await m.shutdown()
            Mentat.reset()

    async def test_ignored_patterns_are_skipped(self, tmp_path, monkeypatch):
        """Files matching watch_ignore patterns should not be indexed."""
        m = _make_e2e_mentat(tmp_path, monkeypatch)

        watch_dir = tmp_path / "project"
        watch_dir.mkdir()

        m.collections_store.create(
            "code",
            watch_paths=[str(watch_dir)],
            watch_ignore=["*.lock", "*.tmp"],
        )

        indexed_paths = []
        original_add = m.add

        async def tracking_add(path, **kwargs):
            indexed_paths.append(path)
            return await original_add(path, **kwargs)

        m.add = tracking_add

        await m.start()

        try:
            await asyncio.sleep(_WATCHER_INIT_DELAY)

            (watch_dir / "pnpm.lock").write_text("lockfile content")
            (watch_dir / "main.py").write_text("print('hello')")

            for _ in range(_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL)
                if any("main.py" in p for p in indexed_paths):
                    break

            assert any("main.py" in p for p in indexed_paths), "main.py should be indexed"
            assert not any("pnpm.lock" in p for p in indexed_paths), "pnpm.lock should be ignored"
        finally:
            await m.shutdown()
            Mentat.reset()

    async def test_dynamic_collection_watch_after_sync(self, tmp_path, monkeypatch):
        """Creating a collection with watch_paths after start, then calling sync()."""
        m = _make_e2e_mentat(tmp_path, monkeypatch)

        watch_dir = tmp_path / "dynamic"
        watch_dir.mkdir()

        await m.start()

        indexed_paths = []
        original_add = m.add

        async def tracking_add(path, **kwargs):
            indexed_paths.append(path)
            return await original_add(path, **kwargs)

        m.add = tracking_add

        try:
            assert len(m.watcher._tasks) == 0

            m.collections_store.create("dynamic", watch_paths=[str(watch_dir)])
            await m.watcher.sync()

            assert "dynamic" in m.watcher._tasks

            # Let the new watcher task initialize
            await asyncio.sleep(_WATCHER_INIT_DELAY)

            (watch_dir / "new.md").write_text("# New file")

            for _ in range(_POLL_ATTEMPTS):
                await asyncio.sleep(_POLL_INTERVAL)
                if any("new.md" in p for p in indexed_paths):
                    break

            assert any("new.md" in p for p in indexed_paths), "File in dynamically watched dir should be indexed"
        finally:
            await m.shutdown()
            Mentat.reset()
