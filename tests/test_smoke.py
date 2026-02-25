import pytest

from mentat.core.hub import Mentat, MentatConfig
from mentat.probes import run_probe
from mentat.probes.base import ProbeResult


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

    def add_stub(self, doc_id, filename, brief_intro, instruction, probe_json):
        self._stubs[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "brief_intro": brief_intro,
            "instruction": instruction,
            "probe_json": probe_json,
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


@pytest.fixture
async def smoke_mentat(tmp_path, monkeypatch):
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


def test_smoke_probe_markdown(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("# Title\n\n## Intro\nhello world")

    result = run_probe(str(p))

    assert isinstance(result, ProbeResult)
    assert result.file_type == "markdown"
    assert len(result.chunks) >= 1


def test_smoke_probe_json(tmp_path):
    p = tmp_path / "sample.json"
    p.write_text('{"name": "mentat", "count": 1}')

    result = run_probe(str(p))

    assert isinstance(result, ProbeResult)
    assert result.file_type == "json"
    assert len(result.chunks) >= 1


def test_smoke_probe_python(tmp_path):
    p = tmp_path / "sample.py"
    p.write_text("def add(a, b):\n    return a + b\n")

    result = run_probe(str(p))

    assert isinstance(result, ProbeResult)
    assert result.file_type in {"python", "code"}
    assert len(result.chunks) >= 1


@pytest.mark.asyncio
async def test_smoke_add_and_search(smoke_mentat, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Orbit\n\nThe moon orbits the earth.")

    doc_id = await smoke_mentat.add(str(p), force=True, wait=True)
    results = await smoke_mentat.search("moon orbit", top_k=5)

    assert isinstance(doc_id, str)
    assert any(r.doc_id == doc_id for r in results)


@pytest.mark.asyncio
async def test_smoke_add_batch(smoke_mentat, tmp_path):
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    p1.write_text("# A\n\nalpha")
    p2.write_text("# B\n\nbeta")

    ids = await smoke_mentat.add_batch([str(p1), str(p2)], force=True, summarize=False)

    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)
    assert smoke_mentat.storage.count_docs() == 2


@pytest.mark.asyncio
async def test_smoke_inspect(smoke_mentat, tmp_path):
    p = tmp_path / "inspect.md"
    p.write_text("# Inspect\n\nOne.\n\n## Two\nMore")

    doc_id = await smoke_mentat.add(str(p), force=True, wait=True)
    info = await smoke_mentat.inspect(doc_id)

    assert info is not None
    assert info["doc_id"] == doc_id
    assert "probe" in info
    assert "chunk_summaries" in info


@pytest.mark.asyncio
async def test_smoke_collections(smoke_mentat, tmp_path):
    p = tmp_path / "col.md"
    p.write_text("# Collection\n\nScoped search content")

    await smoke_mentat.start()
    col = smoke_mentat.collection("team-notes")

    doc_id = await col.add(str(p), force=True, summarize=False)
    done = await smoke_mentat.wait_for_completion(doc_id, timeout=20)
    results = await col.search("scoped search", top_k=5)

    assert done is True
    assert doc_id in col.doc_ids
    assert len(results) >= 1
    assert all(r.doc_id in col.doc_ids for r in results)


@pytest.mark.asyncio
async def test_smoke_stats(smoke_mentat, tmp_path):
    p = tmp_path / "stats.md"
    p.write_text("# Stats\n\ncontent")

    await smoke_mentat.add(str(p), force=True, wait=True)
    s = smoke_mentat.stats()

    assert "docs_indexed" in s
    assert "chunks_stored" in s
    assert "storage_size_bytes" in s
    assert "access_tracker" in s
    assert s["docs_indexed"] >= 1
