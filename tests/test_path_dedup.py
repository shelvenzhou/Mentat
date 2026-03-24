"""Tests for path-based deduplication (PathIndex + add() replacement logic)."""

import pytest
from pathlib import Path

from mentat.storage.cache import PathIndex


# ── PathIndex unit tests ────────────────────────────────────────────────


class TestPathIndex:
    @pytest.fixture
    def index(self, tmp_path):
        return PathIndex(cache_dir=str(tmp_path))

    def test_empty_returns_none(self, index, tmp_path):
        assert index.get(str(tmp_path / "nonexistent.txt")) is None

    def test_put_and_get(self, index, tmp_path):
        p = str(tmp_path / "file.txt")
        index.put(p, "doc-1")
        assert index.get(p) == "doc-1"

    def test_resolves_paths(self, index, tmp_path):
        """Relative and absolute paths to the same file resolve to the same key."""
        abs_path = str(tmp_path / "file.txt")
        index.put(abs_path, "doc-1")
        assert index.get(abs_path) == "doc-1"

    def test_overwrite(self, index, tmp_path):
        p = str(tmp_path / "file.txt")
        index.put(p, "doc-old")
        index.put(p, "doc-new")
        assert index.get(p) == "doc-new"
        assert len(index) == 1

    def test_remove_by_doc_id(self, index, tmp_path):
        p1 = str(tmp_path / "a.txt")
        p2 = str(tmp_path / "b.txt")
        index.put(p1, "doc-1")
        index.put(p2, "doc-2")
        index.remove("doc-1")
        assert index.get(p1) is None
        assert index.get(p2) == "doc-2"

    def test_remove_path(self, index, tmp_path):
        p = str(tmp_path / "file.txt")
        index.put(p, "doc-1")
        index.remove_path(p)
        assert index.get(p) is None
        assert len(index) == 0

    def test_clear(self, index, tmp_path):
        index.put(str(tmp_path / "a.txt"), "doc-1")
        index.put(str(tmp_path / "b.txt"), "doc-2")
        index.clear()
        assert len(index) == 0

    def test_persistence(self, tmp_path):
        idx1 = PathIndex(cache_dir=str(tmp_path))
        p = str(tmp_path / "file.txt")
        idx1.put(p, "doc-1")

        idx2 = PathIndex(cache_dir=str(tmp_path))
        assert idx2.get(p) == "doc-1"

    def test_corrupted_file_starts_fresh(self, tmp_path):
        path = tmp_path / "path_index.json"
        path.write_text("{broken json")

        idx = PathIndex(cache_dir=str(tmp_path))
        assert len(idx) == 0


# ── Integration tests: add() replaces old doc on content change ─────────


@pytest.mark.asyncio
class TestPathDedupIntegration:
    """Verify that re-indexing the same file path with changed content
    replaces the old document instead of creating a duplicate."""

    async def test_same_content_returns_cached(self, mentat_instance, tmp_path):
        """Identical content → cache hit, no new doc created."""
        m = mentat_instance
        f = tmp_path / "hello.md"
        f.write_text("# Hello\nworld")

        id1 = await m.add(str(f), wait=True)
        id2 = await m.add(str(f), wait=True)

        assert id1 == id2
        assert m.storage.count_docs() == 1

    async def test_changed_content_replaces_old_doc(self, mentat_instance, tmp_path):
        """Changed content at the same path → old doc removed, new doc created."""
        m = mentat_instance
        f = tmp_path / "hello.md"

        # Index v1
        f.write_text("# Hello\nversion 1")
        id1 = await m.add(str(f), wait=True)
        assert m.storage.count_docs() == 1

        # Modify file and re-index
        f.write_text("# Hello\nversion 2 with new content")
        id2 = await m.add(str(f), wait=True)

        assert id1 != id2
        # Old doc should be gone — only one doc in storage
        assert m.storage.count_docs() == 1
        assert m.storage.get_stub(id1) is None
        assert m.storage.get_stub(id2) is not None

    async def test_changed_content_old_chunks_removed(self, mentat_instance, tmp_path):
        """Chunks from old version should not remain after replacement."""
        m = mentat_instance
        f = tmp_path / "hello.md"

        f.write_text("# Hello\nversion 1 content here")
        id1 = await m.add(str(f), wait=True)
        assert m.storage.has_chunks(id1)

        f.write_text("# Hello\nversion 2 completely different")
        id2 = await m.add(str(f), wait=True)

        # Old chunks gone, new chunks present
        assert not m.storage.has_chunks(id1)
        assert m.storage.has_chunks(id2)

    async def test_different_paths_not_affected(self, mentat_instance, tmp_path):
        """Different file paths should each keep their own document."""
        m = mentat_instance
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("# File A")
        f2.write_text("# File B")

        id1 = await m.add(str(f1), wait=True)
        id2 = await m.add(str(f2), wait=True)

        assert id1 != id2
        assert m.storage.count_docs() == 2

    async def test_force_reindex_replaces_via_path(self, mentat_instance, tmp_path):
        """force=True with same content still replaces via path index."""
        m = mentat_instance
        f = tmp_path / "hello.md"
        f.write_text("# Hello\nsame content")

        id1 = await m.add(str(f), wait=True)
        id2 = await m.add(str(f), wait=True, force=True)

        assert id1 != id2
        assert m.storage.count_docs() == 1
        assert m.storage.get_stub(id1) is None

    async def test_path_index_updated_on_cache_hit(self, mentat_instance, tmp_path):
        """Cache hit path also populates the path index for legacy docs."""
        m = mentat_instance
        f = tmp_path / "hello.md"
        f.write_text("# Hello")

        id1 = await m.add(str(f), wait=True)
        # Clear path index to simulate legacy doc
        m.path_index.clear()
        assert m.path_index.get(str(f)) is None

        # Second add → cache hit should backfill path index
        id2 = await m.add(str(f), wait=True)
        assert id1 == id2
        assert m.path_index.get(str(f)) == id1


@pytest.mark.asyncio
class TestContentPathDedup:
    """Verify path-based dedup for add_content() (no real file path)."""

    async def test_same_content_returns_cached(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1 = await m.add_content("# Hello\nworld", "readme.md", wait=True)
        id2 = await m.add_content("# Hello\nworld", "readme.md", wait=True)
        assert id1 == id2

    async def test_changed_content_replaces_old(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1 = await m.add_content("# v1", "readme.md", wait=True)
        id2 = await m.add_content("# v2 new stuff", "readme.md", wait=True)

        assert id1 != id2
        assert m.storage.count_docs() == 1
        assert m.storage.get_stub(id1) is None
        assert m.storage.get_stub(id2) is not None

    async def test_different_filenames_not_affected(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1 = await m.add_content("# A", "a.md", wait=True)
        id2 = await m.add_content("# B", "b.md", wait=True)

        assert id1 != id2
        assert m.storage.count_docs() == 2
