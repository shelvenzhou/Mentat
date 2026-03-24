"""Tests for multi-collection search and auto-routing in Hub."""

import pytest
from pathlib import Path

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig


# Reuse conftest fixtures (FakeEmbedding, FakeStorage, mentat_instance)


def _write_file(tmp_path: Path, name: str, content: str) -> str:
    p = tmp_path / name
    p.write_text(content)
    return str(p)


class TestAutoRouting:
    """Test that Hub.add() auto-routes docs to collections based on source."""

    async def test_auto_route_on_add(self, mentat_instance, tmp_path):
        m = mentat_instance
        # Create a collection with auto_add_sources
        m.collections_store.create("files", auto_add_sources=["openclaw:*"])

        path = _write_file(tmp_path, "test.md", "# Hello\nSome content here.")
        doc_id = await m.add(path, source="openclaw:Read", wait=True)

        assert doc_id in m.collections_store.get_doc_ids("files")

    async def test_auto_route_multiple_collections(self, mentat_instance, tmp_path):
        m = mentat_instance
        m.collections_store.create("files", auto_add_sources=["openclaw:*"])
        m.collections_store.create("memory", auto_add_sources=["openclaw:memory"])

        path = _write_file(tmp_path, "note.md", "# Note\nRemember this.")
        doc_id = await m.add(path, source="openclaw:memory", wait=True)

        # Should be in both collections
        assert doc_id in m.collections_store.get_doc_ids("files")
        assert doc_id in m.collections_store.get_doc_ids("memory")

    async def test_explicit_collection_param(self, mentat_instance, tmp_path):
        m = mentat_instance
        m.collections_store.create("ses_abc")

        path = _write_file(tmp_path, "doc.md", "# Doc\nContent.")
        doc_id = await m.add(path, collection="ses_abc", wait=True)

        assert doc_id in m.collections_store.get_doc_ids("ses_abc")

    async def test_explicit_and_auto_route_combined(self, mentat_instance, tmp_path):
        m = mentat_instance
        m.collections_store.create("files", auto_add_sources=["openclaw:*"])
        m.collections_store.create("ses_abc")

        path = _write_file(tmp_path, "doc.md", "# Doc\nContent.")
        doc_id = await m.add(
            path, source="openclaw:Read", collection="ses_abc", wait=True
        )

        assert doc_id in m.collections_store.get_doc_ids("files")
        assert doc_id in m.collections_store.get_doc_ids("ses_abc")

    async def test_auto_route_on_cache_hit(self, mentat_instance, tmp_path):
        m = mentat_instance
        path = _write_file(tmp_path, "doc.md", "# Doc\nContent.")

        # First add — no collection
        doc_id = await m.add(path, wait=True)

        # Create collection after first add
        m.collections_store.create("late", auto_add_sources=["openclaw:*"])

        # Second add — cache hit, but should still route
        doc_id2 = await m.add(path, source="openclaw:Read")
        assert doc_id == doc_id2  # same doc
        assert doc_id in m.collections_store.get_doc_ids("late")

    async def test_no_source_no_auto_route(self, mentat_instance, tmp_path):
        m = mentat_instance
        m.collections_store.create("files", auto_add_sources=["openclaw:*"])

        path = _write_file(tmp_path, "doc.md", "# Doc\nContent.")
        doc_id = await m.add(path, wait=True)

        # No source provided, so no auto-routing
        assert doc_id not in m.collections_store.get_doc_ids("files")


class TestMultiCollectionSearch:
    """Test search with collections parameter."""

    async def _index_docs(self, m, tmp_path):
        """Index 3 docs into different collections."""
        f1 = _write_file(tmp_path, "alpha.md", "# Alpha\nAuthentication middleware code.")
        f2 = _write_file(tmp_path, "beta.md", "# Beta\nDatabase migration scripts.")
        f3 = _write_file(tmp_path, "gamma.md", "# Gamma\nAuthentication notes.")

        id1 = await m.add(f1, wait=True, collection="code")
        id2 = await m.add(f2, wait=True, collection="code")
        id3 = await m.add(f3, wait=True, collection="memory")
        return id1, id2, id3

    async def test_search_single_collection(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1, id2, id3 = await self._index_docs(m, tmp_path)

        results = await m.search("authentication", collections=["code"])
        doc_ids = {r.doc_id for r in results}
        # Should only find docs in "code" collection
        assert id1 in doc_ids
        assert id3 not in doc_ids  # gamma is in "memory"

    async def test_search_multiple_collections(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1, id2, id3 = await self._index_docs(m, tmp_path)

        results = await m.search("authentication", collections=["code", "memory"])
        doc_ids = {r.doc_id for r in results}
        assert id1 in doc_ids
        assert id3 in doc_ids

    async def test_search_empty_collection(self, mentat_instance, tmp_path):
        m = mentat_instance
        m.collections_store.create("empty")
        await self._index_docs(m, tmp_path)

        results = await m.search("authentication", collections=["empty"])
        assert results == []

    async def test_search_nonexistent_collection(self, mentat_instance, tmp_path):
        m = mentat_instance
        await self._index_docs(m, tmp_path)

        results = await m.search("authentication", collections=["nope"])
        assert results == []

    async def test_search_no_collections_is_global(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1, id2, id3 = await self._index_docs(m, tmp_path)

        results = await m.search("authentication")
        doc_ids = {r.doc_id for r in results}
        # Global search — should find both alpha and gamma
        assert id1 in doc_ids
        assert id3 in doc_ids

    async def test_search_grouped_with_collections(self, mentat_instance, tmp_path):
        m = mentat_instance
        id1, id2, id3 = await self._index_docs(m, tmp_path)

        results = await m.search_grouped("authentication", collections=["code"])
        doc_ids = {r.doc_id for r in results}
        assert id1 in doc_ids
        assert id3 not in doc_ids


class TestCollectionClassIntegration:
    """Test the Collection wrapper class with the new routing."""

    async def test_collection_add_routes_to_collection(self, mentat_instance, tmp_path):
        m = mentat_instance
        coll = m.collection("papers")

        path = _write_file(tmp_path, "paper.md", "# Paper\nResearch content.")
        doc_id = await coll.add(path)

        assert doc_id in m.collections_store.get_doc_ids("papers")

    async def test_collection_search_scoped(self, mentat_instance, tmp_path):
        m = mentat_instance

        f1 = _write_file(tmp_path, "a.md", "# A\nAuth code.")
        f2 = _write_file(tmp_path, "b.md", "# B\nAuth notes.")

        # Use wait=True + collection to avoid background processor dependency
        id1 = await m.add(f1, wait=True, collection="code")
        id2 = await m.add(f2, wait=True, collection="memory")

        coll_code = m.collection("code")
        results = await coll_code.search("auth")
        doc_ids = {r.doc_id for r in results}
        assert id1 in doc_ids
        assert id2 not in doc_ids
