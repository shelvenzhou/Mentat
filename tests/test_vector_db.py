"""Tests for LanceDB storage (mentat/storage/vector_db.py).

Uses a real LanceDB instance in a tmp directory — no mocking needed.
"""

import pytest
from mentat.storage.vector_db import LanceDBStorage


@pytest.fixture
def db(tmp_path):
    return LanceDBStorage(db_path=str(tmp_path / "test_db"))


def _vec(val: float = 0.1):
    """Create a 4-dim test vector."""
    return [val, val, val, val]


class TestStubs:
    def test_add_and_get(self, db):
        db.add_stub("d1", "file.pdf", "A brief intro", "Read section 3", "{}")
        stub = db.get_stub("d1")
        assert stub is not None
        assert stub["filename"] == "file.pdf"
        assert stub["brief_intro"] == "A brief intro"
        assert stub["instruction"] == "Read section 3"

    def test_get_nonexistent(self, db):
        assert db.get_stub("nope") is None

    def test_count_docs(self, db):
        assert db.count_docs() == 0
        db.add_stub("d1", "a.pdf", "", "", "{}")
        db.add_stub("d2", "b.pdf", "", "", "{}")
        assert db.count_docs() == 2


class TestChunks:
    def test_add_and_search(self, db):
        db.add_chunks(
            [
                {
                    "chunk_id": "d1_0",
                    "doc_id": "d1",
                    "filename": "file.pdf",
                    "content": "Some content about AI",
                    "summary": "Content about AI",
                    "section": "Intro",
                    "chunk_index": 0,
                    "vector": _vec(0.9),
                }
            ]
        )
        results = db.search(_vec(0.9), limit=1)
        assert len(results) == 1
        assert results[0]["content"] == "Some content about AI"
        assert results[0]["summary"] == "Content about AI"
        assert results[0]["section"] == "Intro"

    def test_count_chunks(self, db):
        assert db.count_chunks() == 0
        db.add_chunks(
            [
                {
                    "chunk_id": "d1_0",
                    "doc_id": "d1",
                    "filename": "f.pdf",
                    "content": "c",
                    "summary": "s",
                    "section": "",
                    "chunk_index": 0,
                    "vector": _vec(),
                },
                {
                    "chunk_id": "d1_1",
                    "doc_id": "d1",
                    "filename": "f.pdf",
                    "content": "c2",
                    "summary": "s2",
                    "section": "",
                    "chunk_index": 1,
                    "vector": _vec(),
                },
            ]
        )
        assert db.count_chunks() == 2

    def test_search_with_doc_filter(self, db):
        for doc_id in ("d1", "d2"):
            db.add_chunks(
                [
                    {
                        "chunk_id": f"{doc_id}_0",
                        "doc_id": doc_id,
                        "filename": f"{doc_id}.pdf",
                        "content": f"Content from {doc_id}",
                        "summary": f"Summary from {doc_id}",
                        "section": "",
                        "chunk_index": 0,
                        "vector": _vec(),
                    }
                ]
            )
        # Search only d1
        results = db.search(_vec(), limit=10, doc_ids=["d1"])
        assert all(r["doc_id"] == "d1" for r in results)

    def test_get_chunks_by_doc(self, db):
        db.add_chunks(
            [
                {
                    "chunk_id": "d1_0",
                    "doc_id": "d1",
                    "filename": "f.pdf",
                    "content": "c0",
                    "summary": "s0",
                    "section": "A",
                    "chunk_index": 0,
                    "vector": _vec(),
                },
                {
                    "chunk_id": "d1_1",
                    "doc_id": "d1",
                    "filename": "f.pdf",
                    "content": "c1",
                    "summary": "s1",
                    "section": "B",
                    "chunk_index": 1,
                    "vector": _vec(),
                },
                {
                    "chunk_id": "d2_0",
                    "doc_id": "d2",
                    "filename": "g.pdf",
                    "content": "other",
                    "summary": "other",
                    "section": "",
                    "chunk_index": 0,
                    "vector": _vec(),
                },
            ]
        )
        rows = db.get_chunks_by_doc("d1")
        assert len(rows) == 2
        assert rows[0]["chunk_index"] == 0
        assert rows[1]["chunk_index"] == 1
        assert rows[0]["summary"] == "s0"

    def test_get_chunks_by_doc_nonexistent(self, db):
        assert db.get_chunks_by_doc("nope") == []

    def test_list_docs(self, db):
        db.add_stub("d1", "a.pdf", "intro1", "inst1", "{}")
        db.add_stub("d2", "b.csv", "intro2", "inst2", "{}")
        docs = db.list_docs()
        filenames = {d["filename"] for d in docs}
        assert filenames == {"a.pdf", "b.csv"}
