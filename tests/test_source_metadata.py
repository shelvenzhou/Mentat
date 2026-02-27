"""Tests for source/metadata integration in the Mentat hub.

Uses FakeStorage and FakeEmbedding to validate that source and metadata
fields propagate correctly through add, search, inspect, and add_content.
"""

import pytest

from mentat.core.hub import Mentat, MentatConfig


class FakeEmbedding:
    def _vec(self, text: str):
        base = sum(ord(c) for c in text) % 97
        return [float((base + i) % 17) / 17.0 for i in range(8)]

    async def embed(self, text: str):
        return self._vec(text)

    async def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


class FakeStorage:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path
        self._stubs = {}
        self._chunks = []

    def add_stub(self, doc_id, filename, brief_intro, instruction, probe_json,
                 source="", metadata_json="{}"):
        self._stubs[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "brief_intro": brief_intro,
            "instruction": instruction,
            "probe_json": probe_json,
            "source": source,
            "metadata_json": metadata_json,
        }

    def get_stub(self, doc_id):
        return self._stubs.get(doc_id)

    def _ensure_chunks_table(self, vector_dim: int):
        return None

    def add_chunks(self, chunks):
        self._chunks.extend(chunks)

    def search(self, query_vector, query_text="", limit=5, use_hybrid=False, doc_ids=None):
        rows = self._chunks
        if doc_ids is not None:
            rows = [r for r in rows if r.get("doc_id") in set(doc_ids)]

        def _dist(row):
            vec = row.get("vector", [])
            return sum((a - b) ** 2 for a, b in zip(query_vector, vec))

        ranked = sorted(rows, key=_dist)[:limit]
        out = []
        for row in ranked:
            r = dict(row)
            r["_distance"] = _dist(row)
            out.append(r)
        return out

    def get_chunks_by_doc(self, doc_id):
        rows = [r for r in self._chunks if r.get("doc_id") == doc_id]
        rows.sort(key=lambda r: r.get("chunk_index", 0))
        return rows

    def update_chunks(self, doc_id, updated_rows):
        self._chunks = [r for r in self._chunks if r.get("doc_id") != doc_id]
        self._chunks.extend(updated_rows)

    def count_docs(self):
        return len(self._stubs)

    def count_chunks(self):
        return len(self._chunks)

    def has_chunks(self, doc_id):
        return any(r.get("doc_id") == doc_id for r in self._chunks)

    def list_docs(self):
        return list(self._stubs.values())

    def get_doc_ids_by_source(self, source):
        results = []
        for doc_id, stub in self._stubs.items():
            s = stub.get("source", "")
            if source.endswith("*"):
                if s.startswith(source[:-1]):
                    results.append(doc_id)
            elif s == source:
                results.append(doc_id)
        return results


@pytest.fixture
async def mentat_with_source(tmp_path, monkeypatch):
    monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
    cfg = MentatConfig(
        db_path=str(tmp_path / "db"),
        storage_dir=str(tmp_path / "files"),
        max_concurrent_tasks=2,
    )
    m = Mentat(cfg)
    m.embeddings = FakeEmbedding()
    yield m
    await m.shutdown()
    Mentat.reset()


# ── add() with source/metadata ──────────────────────────────────────


@pytest.mark.asyncio
async def test_add_with_source(mentat_with_source, tmp_path):
    p = tmp_path / "page.md"
    p.write_text("# Web Page\n\nContent from web.")

    doc_id = await mentat_with_source.add(
        str(p), force=True, wait=True, source="web_fetch"
    )

    stub = mentat_with_source.storage.get_stub(doc_id)
    assert stub is not None
    assert stub["source"] == "web_fetch"


@pytest.mark.asyncio
async def test_add_with_metadata(mentat_with_source, tmp_path):
    p = tmp_path / "email.md"
    p.write_text("# Email\n\nHello from Gmail.")

    doc_id = await mentat_with_source.add(
        str(p), force=True, wait=True,
        source="composio:gmail",
        metadata={"subject": "Hello", "from": "test@example.com"},
    )

    stub = mentat_with_source.storage.get_stub(doc_id)
    assert stub is not None
    assert stub["source"] == "composio:gmail"
    assert '"subject": "Hello"' in stub["metadata_json"]


@pytest.mark.asyncio
async def test_add_default_source_is_empty(mentat_with_source, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Doc\n\nPlain content.")

    doc_id = await mentat_with_source.add(str(p), force=True, wait=True)

    stub = mentat_with_source.storage.get_stub(doc_id)
    assert stub["source"] == ""
    assert stub["metadata_json"] == "{}"


# ── search() with source filter ─────────────────────────────────────


@pytest.mark.asyncio
async def test_search_with_source_filter(mentat_with_source, tmp_path):
    """Source filter should restrict results to matching docs."""
    p1 = tmp_path / "web.md"
    p1.write_text("# Web Page\n\nMoon orbits earth.")
    p2 = tmp_path / "email.md"
    p2.write_text("# Email\n\nMoon facts from space agency.")

    d1 = await mentat_with_source.add(str(p1), force=True, wait=True, source="web_fetch")
    d2 = await mentat_with_source.add(str(p2), force=True, wait=True, source="composio:gmail")

    # Search only web_fetch
    results = await mentat_with_source.search("moon", top_k=10, source="web_fetch")
    doc_ids = {r.doc_id for r in results}
    assert d1 in doc_ids
    assert d2 not in doc_ids


@pytest.mark.asyncio
async def test_search_with_glob_source_filter(mentat_with_source, tmp_path):
    """Glob source like 'composio:*' should match all composio sources."""
    p1 = tmp_path / "gmail.md"
    p1.write_text("# Gmail\n\nEmail content about stars.")
    p2 = tmp_path / "notion.md"
    p2.write_text("# Notion\n\nNotion page about stars.")
    p3 = tmp_path / "web.md"
    p3.write_text("# Web\n\nWeb page about stars.")

    d1 = await mentat_with_source.add(str(p1), force=True, wait=True, source="composio:gmail")
    d2 = await mentat_with_source.add(str(p2), force=True, wait=True, source="composio:notion")
    d3 = await mentat_with_source.add(str(p3), force=True, wait=True, source="web_fetch")

    results = await mentat_with_source.search("stars", top_k=10, source="composio:*")
    doc_ids = {r.doc_id for r in results}
    assert d1 in doc_ids
    assert d2 in doc_ids
    assert d3 not in doc_ids


@pytest.mark.asyncio
async def test_search_source_no_match_returns_empty(mentat_with_source, tmp_path):
    """When no docs match the source filter, return empty list."""
    p = tmp_path / "doc.md"
    p.write_text("# Doc\n\nSome content.")

    await mentat_with_source.add(str(p), force=True, wait=True, source="web_fetch")

    results = await mentat_with_source.search("content", top_k=10, source="browser")
    assert results == []


@pytest.mark.asyncio
async def test_search_without_source_returns_all(mentat_with_source, tmp_path):
    """Without source filter, search returns all matching docs."""
    p1 = tmp_path / "a.md"
    p1.write_text("# A\n\nApple content.")
    p2 = tmp_path / "b.md"
    p2.write_text("# B\n\nApple orchard.")

    d1 = await mentat_with_source.add(str(p1), force=True, wait=True, source="web_fetch")
    d2 = await mentat_with_source.add(str(p2), force=True, wait=True, source="composio:gmail")

    results = await mentat_with_source.search("apple", top_k=10)
    doc_ids = {r.doc_id for r in results}
    assert d1 in doc_ids
    assert d2 in doc_ids


# ── inspect() returns source/metadata ───────────────────────────────


@pytest.mark.asyncio
async def test_inspect_includes_source(mentat_with_source, tmp_path):
    p = tmp_path / "page.md"
    p.write_text("# Page\n\nContent.")

    doc_id = await mentat_with_source.add(
        str(p), force=True, wait=True,
        source="browser",
        metadata={"url": "https://example.com"},
    )

    info = await mentat_with_source.inspect(doc_id)
    assert info is not None
    assert info["source"] == "browser"
    assert info["metadata"]["url"] == "https://example.com"


# ── add_content() with source/metadata ──────────────────────────────


@pytest.mark.asyncio
async def test_add_content_with_source(mentat_with_source):
    doc_id = await mentat_with_source.add_content(
        content="# Hello\n\nThis is a web page about planets.",
        filename="planets.md",
        source="web_fetch",
        metadata={"url": "https://example.com/planets"},
        wait=True,
    )

    stub = mentat_with_source.storage.get_stub(doc_id)
    assert stub is not None
    assert stub["source"] == "web_fetch"


@pytest.mark.asyncio
async def test_add_content_composio_source(mentat_with_source):
    doc_id = await mentat_with_source.add_content(
        content="# Meeting Notes\n\nDiscussion about project timeline.",
        filename="meeting-notes.md",
        source="composio:gmail",
        metadata={"subject": "Meeting Notes", "from": "alice@example.com"},
        wait=True,
    )

    stub = mentat_with_source.storage.get_stub(doc_id)
    assert stub["source"] == "composio:gmail"

    # Should be findable by source
    results = await mentat_with_source.search(
        "meeting", top_k=5, source="composio:gmail"
    )
    assert any(r.doc_id == doc_id for r in results)


# ── search toc_only with source ─────────────────────────────────────


@pytest.mark.asyncio
async def test_toc_only_search_with_source(mentat_with_source, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Chapter 1\n\nIntro.\n\n## Setup\n\nSetup steps.")

    doc_id = await mentat_with_source.add(
        str(p), force=True, wait=True, source="web_fetch"
    )

    results = await mentat_with_source.search(
        "setup", top_k=5, toc_only=True, source="web_fetch"
    )
    assert len(results) >= 1
    assert results[0].content == ""  # toc_only suppresses content
    assert results[0].source == "web_fetch"


# ── MentatResult source field ───────────────────────────────────────


@pytest.mark.asyncio
async def test_search_result_includes_source(mentat_with_source, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Test\n\nUnique rocket launch content.")

    await mentat_with_source.add(
        str(p), force=True, wait=True, source="browser"
    )

    results = await mentat_with_source.search("rocket launch", top_k=5)
    assert len(results) >= 1
    assert results[0].source == "browser"
