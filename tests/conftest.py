"""Shared test fixtures for the Mentat test suite.

Provides FakeEmbedding, FakeStorage, and a reusable ``mentat_instance`` fixture
so individual test files don't need to duplicate boilerplate.
"""

import pytest

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig


class FakeEmbedding:
    """Deterministic embedding provider for tests (no API calls)."""

    def _vec(self, text: str):
        base = sum(ord(c) for c in text) % 97
        return [float((base + i) % 17) / 17.0 for i in range(8)]

    async def embed(self, text: str):
        return self._vec(text)

    async def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


class FakeStorage:
    """In-memory storage backend for tests (no LanceDB dependency)."""

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

    def delete_doc(self, doc_id):
        self._stubs.pop(doc_id, None)
        self._chunks = [r for r in self._chunks if r.get("doc_id") != doc_id]

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
async def mentat_instance(tmp_path, monkeypatch):
    """A Mentat instance backed by FakeStorage and FakeEmbedding."""
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
