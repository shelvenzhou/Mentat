"""Tests for the file watcher module."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mentat.core.watcher import MentatWatcher, _sha256, _make_filter


# ── Unit tests for helpers ──────────────────────────────────────────────


class TestSha256:
    def test_hash_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h = _sha256(str(f))
        assert h is not None
        assert len(h) == 64  # hex SHA-256

    def test_hash_changes_with_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("version 1")
        h1 = _sha256(str(f))
        f.write_text("version 2")
        h2 = _sha256(str(f))
        assert h1 != h2

    def test_hash_nonexistent_returns_none(self):
        assert _sha256("/nonexistent/path") is None


class TestMakeFilter:
    def test_allows_normal_files(self):
        filt = _make_filter(["node_modules", "*.lock"])
        assert filt(None, "/src/main.py") is True

    def test_blocks_glob_pattern(self):
        filt = _make_filter(["*.lock"])
        assert filt(None, "/src/package-lock.json") is True  # doesn't match *.lock
        assert filt(None, "/src/pnpm.lock") is False

    def test_blocks_directory_name(self):
        filt = _make_filter(["node_modules"])
        assert filt(None, "/project/node_modules/foo/bar.js") is False
        assert filt(None, "/project/src/main.js") is True

    def test_empty_ignore_allows_all(self):
        filt = _make_filter([])
        assert filt(None, "/anything.txt") is True


# ── Watcher lifecycle tests ─────────────────────────────────────────────


class TestWatcherSync:
    """Test that sync() starts/stops tasks based on collection watch configs."""

    async def test_sync_starts_tasks_for_watched_collections(self, tmp_path):
        mentat = MagicMock()
        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()

        mentat.collections_store.get_all_watch_configs.return_value = {
            "code": {"watch_paths": [str(watch_dir)], "watch_ignore": []},
        }

        watcher = MentatWatcher(mentat)
        watcher._running = True

        # Patch _watch_collection to avoid actual file watching
        with patch.object(watcher, "_watch_collection", new_callable=AsyncMock):
            await watcher.sync()
            assert "code" in watcher._tasks

    async def test_sync_stops_removed_collections(self, tmp_path):
        mentat = MagicMock()
        watch_dir = tmp_path / "watched"
        watch_dir.mkdir()

        mentat.collections_store.get_all_watch_configs.return_value = {
            "code": {"watch_paths": [str(watch_dir)], "watch_ignore": []},
        }

        watcher = MentatWatcher(mentat)
        watcher._running = True

        with patch.object(watcher, "_watch_collection", new_callable=AsyncMock):
            await watcher.sync()
            assert "code" in watcher._tasks

            # Remove collection from configs
            mentat.collections_store.get_all_watch_configs.return_value = {}
            await watcher.sync()
            assert "code" not in watcher._tasks

    async def test_stop_cancels_all_tasks(self, tmp_path):
        mentat = MagicMock()
        watcher = MentatWatcher(mentat)
        watcher._running = True

        # Add a fake task
        async def fake_watch():
            await asyncio.sleep(100)

        watcher._tasks["code"] = asyncio.create_task(fake_watch())
        await watcher.stop()
        assert watcher._tasks == {}
        assert watcher._running is False


class TestInitialScan:
    """Test that _initial_scan indexes existing files."""

    async def test_scans_existing_files(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        (tmp_path / "a.md").write_text("file a")
        (tmp_path / "b.txt").write_text("file b")

        await watcher._initial_scan("notes", [str(tmp_path)], [])

        assert mentat.add.call_count == 2
        # All calls use force=False (cache-aware)
        for call in mentat.add.call_args_list:
            assert call.kwargs["force"] is False
            assert call.kwargs["collection"] == "notes"
            assert call.kwargs["source"] == "watcher:notes"

    async def test_scans_subdirectories(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-456")

        watcher = MentatWatcher(mentat)

        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("top")
        (sub / "nested.md").write_text("nested")

        await watcher._initial_scan("code", [str(tmp_path)], [])
        assert mentat.add.call_count == 2

    async def test_respects_ignore_patterns(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-789")

        watcher = MentatWatcher(mentat)

        (tmp_path / "main.py").write_text("print('hi')")
        (tmp_path / "pnpm.lock").write_text("lockfile")
        (tmp_path / "data.tmp").write_text("temp")

        await watcher._initial_scan("code", [str(tmp_path)], ["*.lock", "*.tmp"])

        assert mentat.add.call_count == 1
        mentat.add.assert_called_once_with(
            str(tmp_path / "main.py"),
            force=False,
            source="watcher:code",
            collection="code",
        )

    async def test_seeds_hashes(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        f = tmp_path / "note.md"
        f.write_text("content")

        await watcher._initial_scan("memory", [str(tmp_path)], [])

        # Hash should be seeded so _handle_change skips unchanged files
        assert str(f) in watcher._hashes
        assert watcher._hashes[str(f)] == _sha256(str(f))

    async def test_skips_empty_directory(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock()

        watcher = MentatWatcher(mentat)
        await watcher._initial_scan("empty", [str(tmp_path)], [])

        mentat.add.assert_not_called()


class TestWatcherHandleChange:
    """Test the _handle_change method directly."""

    async def test_indexes_changed_file(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        f = tmp_path / "test.md"
        f.write_text("# Hello\nContent.")

        await watcher._handle_change("code", str(f))

        mentat.add.assert_called_once_with(
            str(f),
            force=True,
            source="watcher:code",
            collection="code",
        )

    async def test_throttles_rapid_changes(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        f = tmp_path / "test.md"
        f.write_text("# Hello\nContent.")

        await watcher._handle_change("code", str(f))
        await watcher._handle_change("code", str(f))  # should be throttled

        assert mentat.add.call_count == 1

    async def test_skips_unchanged_content(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        f = tmp_path / "test.md"
        f.write_text("# Hello\nContent.")

        await watcher._handle_change("code", str(f))

        # Reset throttle to allow re-processing
        watcher._throttle.clear()

        # Same content — hash unchanged
        await watcher._handle_change("code", str(f))

        assert mentat.add.call_count == 1

    async def test_reindexes_on_content_change(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock(return_value="doc-123")

        watcher = MentatWatcher(mentat)

        f = tmp_path / "test.md"
        f.write_text("version 1")
        await watcher._handle_change("code", str(f))

        # Clear throttle and change content
        watcher._throttle.clear()
        f.write_text("version 2")
        await watcher._handle_change("code", str(f))

        assert mentat.add.call_count == 2

    async def test_skips_nonexistent_file(self, tmp_path):
        mentat = MagicMock()
        mentat.add = AsyncMock()

        watcher = MentatWatcher(mentat)
        await watcher._handle_change("code", str(tmp_path / "missing.txt"))

        mentat.add.assert_not_called()
