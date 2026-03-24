import asyncio
from unittest.mock import AsyncMock

import pytest

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig


class FakeEmbedding:
    def __init__(self, delay: float = 0.0):
        self.delay = delay

    def _vec(self, text: str):
        base = sum(ord(c) for c in text) % 101
        return [float((base + i) % 19) / 19.0 for i in range(8)]

    async def embed(self, text: str):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self._vec(text)

    async def embed_batch(self, texts):
        if self.delay:
            await asyncio.sleep(self.delay)
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
async def async_mentat(tmp_path, monkeypatch):
    monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
    cfg = MentatConfig(
        db_path=str(tmp_path / "db"),
        storage_dir=str(tmp_path / "files"),
        max_concurrent_tasks=3,
    )
    m = Mentat(cfg)
    m.embeddings = FakeEmbedding(delay=0.05)

    async def _summaries(pr):
        await asyncio.sleep(0.05)
        return [f"summary-{pr.filename}-{i}" for i, _ in enumerate(pr.chunks)]

    m.librarian.summarize_chunks = AsyncMock(side_effect=_summaries)
    yield m
    await m.shutdown()
    Mentat.reset()


@pytest.mark.asyncio
async def test_async_summary_flag_propagates(async_mentat, tmp_path):
    p = tmp_path / "flag.md"
    p.write_text("# Flag\n\ncontent")

    await async_mentat.start()
    doc_id = await async_mentat.add(str(p), force=True, summarize=True, wait=False)

    status = async_mentat.processor.queue.get_status(doc_id)
    assert status is not None
    assert status["needs_summarization"] is True


@pytest.mark.asyncio
async def test_async_summary_produces_summaries(async_mentat, tmp_path):
    p = tmp_path / "sum.md"
    p.write_text("# Sum\n\ncontent one\n\n## Two\ncontent two")

    await async_mentat.start()
    doc_id = await async_mentat.add(str(p), force=True, summarize=True, wait=False)
    done = await async_mentat.wait_for_completion(doc_id, timeout=20)

    info = await async_mentat.inspect(doc_id, full=True)

    assert done is True
    assert info is not None
    assert "chunk_summaries" in info
    assert all(row["summary"].startswith("summary-") for row in info["chunk_summaries"])
    assert async_mentat.librarian.summarize_chunks.await_count >= 1


@pytest.mark.asyncio
async def test_async_summary_skip_when_false(async_mentat, tmp_path):
    p = tmp_path / "nosum.md"
    p.write_text("# No summary\n\ncontent")

    await async_mentat.start()
    doc_id = await async_mentat.add(str(p), force=True, summarize=False, wait=False)
    done = await async_mentat.wait_for_completion(doc_id, timeout=20)

    assert done is True
    assert async_mentat.librarian.summarize_chunks.await_count == 0


@pytest.mark.asyncio
async def test_async_summary_concurrent_docs(async_mentat, tmp_path):
    await async_mentat.start()

    paths = []
    for i in range(4):
        p = tmp_path / f"doc_{i}.md"
        p.write_text(f"# Doc {i}\n\ncontent {i}")
        paths.append(str(p))

    doc_ids = await asyncio.gather(*[
        async_mentat.add(path, force=True, summarize=True, wait=False)
        for path in paths
    ])

    done_flags = await asyncio.gather(*[
        async_mentat.wait_for_completion(d, timeout=20) for d in doc_ids
    ])

    infos = await asyncio.gather(*[async_mentat.inspect(d, full=True) for d in doc_ids])

    assert all(done_flags)
    assert all(info is not None for info in infos)
    assert all(info.get("chunk_summaries") for info in infos)


@pytest.mark.asyncio
async def test_async_summary_status_progression(async_mentat, tmp_path):
    p = tmp_path / "status.md"
    p.write_text("# Status\n\nA\n\n## B\nC")

    await async_mentat.start()
    doc_id = await async_mentat.add(str(p), force=True, summarize=True, wait=False)

    first = async_mentat.get_processing_status(doc_id)["status"]
    assert first in {"pending", "processing", "completed"}

    saw_processing = False
    deadline = asyncio.get_event_loop().time() + 3
    while asyncio.get_event_loop().time() < deadline:
        s = async_mentat.get_processing_status(doc_id)["status"]
        if s == "processing":
            saw_processing = True
            break
        if s == "completed":
            break
        await asyncio.sleep(0.05)

    done = await async_mentat.wait_for_completion(doc_id, timeout=20)
    final_status = async_mentat.get_processing_status(doc_id)["status"]

    assert done is True
    assert final_status == "completed"
    assert saw_processing or first in {"processing", "completed"}
