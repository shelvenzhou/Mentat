"""Smoke tests for append-mode watcher.

These test the watcher's offset tracking and incremental indexing logic
in isolation (no real Mentat instance needed — uses mocks).
"""

import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mentat.core.watcher import MentatWatcher


@pytest.fixture
def mock_mentat(tmp_path):
    """Create a mock Mentat hub with the methods watcher needs."""
    m = MagicMock()
    m.config.storage_dir = str(tmp_path / "storage")
    m.config.db_path = str(tmp_path / "db")
    Path(m.config.storage_dir).mkdir()
    Path(m.config.db_path).mkdir()
    m.add = AsyncMock(return_value="doc-123")
    m.collections_store = MagicMock()
    return m


@pytest.fixture
def sessions_dir(tmp_path):
    d = tmp_path / "sessions"
    d.mkdir()
    return d


def write_jsonl(path, lines):
    """Append JSONL lines to a file."""
    with open(path, "a") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


class TestAppendModeOffsets:
    @pytest.mark.asyncio
    async def test_initial_index_from_zero(self, mock_mentat, sessions_dir):
        """New file with no offset → processed from byte 0."""
        mock_mentat.collections_store.get_all_watch_configs.return_value = {
            "chat_history": {
                "watch_paths": [str(sessions_dir)],
                "watch_ignore": [],
                "watch_mode": "append",
            }
        }

        session_file = sessions_dir / "test-session.jsonl"
        write_jsonl(session_file, [
            {"type": "message", "id": "1"},
            {"type": "message", "id": "2"},
        ])

        watcher = MentatWatcher(mock_mentat)
        watcher._configs = mock_mentat.collections_store.get_all_watch_configs()

        await watcher._handle_append_change("chat_history", str(session_file))

        mock_mentat.add.assert_called_once()
        call_kwargs = mock_mentat.add.call_args
        assert call_kwargs.kwargs["_logical_filename"] == "test-session.jsonl@0"
        assert call_kwargs.kwargs["metadata"]["session_id"] == "test-session"
        assert call_kwargs.kwargs["metadata"]["offset"] == 0

        # Offset should be updated
        assert watcher._offsets[str(session_file)] > 0

    @pytest.mark.asyncio
    async def test_incremental_after_append(self, mock_mentat, sessions_dir):
        """Appending to file → only new bytes processed."""
        mock_mentat.collections_store.get_all_watch_configs.return_value = {
            "chat_history": {
                "watch_paths": [str(sessions_dir)],
                "watch_ignore": [],
                "watch_mode": "append",
            }
        }

        session_file = sessions_dir / "inc-session.jsonl"
        write_jsonl(session_file, [{"type": "message", "id": "1"}])

        watcher = MentatWatcher(mock_mentat)
        watcher._configs = mock_mentat.collections_store.get_all_watch_configs()

        # First index
        await watcher._handle_append_change("chat_history", str(session_file))
        first_offset = watcher._offsets[str(session_file)]
        assert first_offset > 0

        mock_mentat.add.reset_mock()
        watcher._throttle.clear()  # Reset throttle

        # Append more
        write_jsonl(session_file, [{"type": "message", "id": "2"}])

        await watcher._handle_append_change("chat_history", str(session_file))

        mock_mentat.add.assert_called_once()
        call_kwargs = mock_mentat.add.call_args
        # Second call should use the first offset as start
        assert call_kwargs.kwargs["_logical_filename"] == f"inc-session.jsonl@{first_offset}"
        assert call_kwargs.kwargs["metadata"]["offset"] == first_offset

    @pytest.mark.asyncio
    async def test_no_growth_skipped(self, mock_mentat, sessions_dir):
        """File same size as offset → no indexing."""
        mock_mentat.collections_store.get_all_watch_configs.return_value = {
            "chat_history": {
                "watch_paths": [str(sessions_dir)],
                "watch_ignore": [],
                "watch_mode": "append",
            }
        }

        session_file = sessions_dir / "static.jsonl"
        write_jsonl(session_file, [{"type": "message", "id": "1"}])

        watcher = MentatWatcher(mock_mentat)
        watcher._configs = mock_mentat.collections_store.get_all_watch_configs()

        # Set offset to file size (pretend already indexed)
        watcher._offsets[str(session_file)] = session_file.stat().st_size

        await watcher._handle_append_change("chat_history", str(session_file))

        mock_mentat.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_deleted_clears_offset(self, mock_mentat, sessions_dir):
        """Deleted file → offset cleared, no error."""
        watcher = MentatWatcher(mock_mentat)
        watcher._configs = {
            "chat_history": {"watch_paths": [str(sessions_dir)], "watch_ignore": [], "watch_mode": "append"}
        }

        fake_path = str(sessions_dir / "gone.jsonl")
        watcher._offsets[fake_path] = 100

        await watcher._handle_append_change("chat_history", fake_path)

        assert fake_path not in watcher._offsets
        mock_mentat.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_probe_config_passed(self, mock_mentat, sessions_dir):
        """watch_probe_config is forwarded to add()."""
        probe_cfg = {"filters": [{"field": "type", "op": "eq", "value": "message"}]}
        mock_mentat.collections_store.get_all_watch_configs.return_value = {
            "chat_history": {
                "watch_paths": [str(sessions_dir)],
                "watch_ignore": [],
                "watch_mode": "append",
                "watch_probe_config": probe_cfg,
            }
        }

        session_file = sessions_dir / "configured.jsonl"
        write_jsonl(session_file, [{"type": "message", "id": "1"}])

        watcher = MentatWatcher(mock_mentat)
        watcher._configs = mock_mentat.collections_store.get_all_watch_configs()

        await watcher._handle_append_change("chat_history", str(session_file))

        call_kwargs = mock_mentat.add.call_args
        assert call_kwargs.kwargs["probe_config"] == probe_cfg


class TestOffsetPersistence:
    def test_save_and_load(self, mock_mentat, tmp_path):
        """Offsets survive save/load cycle."""
        watcher = MentatWatcher(mock_mentat)
        watcher._offsets = {"/path/a.jsonl": 100, "/path/b.jsonl": 200}

        watcher._save_offsets()
        assert watcher._offsets_path().exists()

        watcher2 = MentatWatcher(mock_mentat)
        watcher2._load_offsets()
        assert watcher2._offsets == {"/path/a.jsonl": 100, "/path/b.jsonl": 200}

    def test_missing_file_starts_empty(self, mock_mentat):
        """No offsets file → empty dict, no error."""
        watcher = MentatWatcher(mock_mentat)
        watcher._load_offsets()
        assert watcher._offsets == {}


class TestIncompleteLineHandling:
    @pytest.mark.asyncio
    async def test_partial_last_line_discarded(self, mock_mentat, sessions_dir):
        """Incomplete last line is not indexed; offset excludes it."""
        mock_mentat.collections_store.get_all_watch_configs.return_value = {
            "chat_history": {
                "watch_paths": [str(sessions_dir)],
                "watch_ignore": [],
                "watch_mode": "append",
            }
        }

        session_file = sessions_dir / "partial.jsonl"
        # Write one complete line + one partial
        with open(session_file, "w") as f:
            f.write('{"complete": true}\n{"incomple')

        watcher = MentatWatcher(mock_mentat)
        watcher._configs = mock_mentat.collections_store.get_all_watch_configs()

        await watcher._handle_append_change("chat_history", str(session_file))

        mock_mentat.add.assert_called_once()
        # Offset should only cover the complete line
        offset = watcher._offsets[str(session_file)]
        assert offset == len('{"complete": true}\n')
