"""HTTP endpoint tests for the Mentat FastAPI server."""

import pytest
from httpx import AsyncClient, ASGITransport

from mentat.server import create_app
from mentat.core.hub import Mentat, MentatConfig
from tests.conftest import FakeStorage, FakeEmbedding


@pytest.fixture
async def http_client(tmp_path, monkeypatch):
    """AsyncClient wired to a test FastAPI app with fake storage."""
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


# ── Health ───────────────────────────────────────────────────────────


async def test_health(http_client):
    resp = await http_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Index + Search ───────────────────────────────────────────────────


async def test_index_and_search(http_client, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Moon\n\nThe moon orbits the earth.")

    # Index
    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    assert resp.status_code == 200
    data = resp.json()
    assert "doc_id" in data
    doc_id = data["doc_id"]

    # Search
    resp = await http_client.post("/search", json={"query": "moon orbit", "top_k": 5})
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    assert any(r["doc_id"] == doc_id for r in results)


# ── Search Grouped ───────────────────────────────────────────────────


async def test_search_grouped_endpoint(http_client, tmp_path):
    p = tmp_path / "grouped.md"
    p.write_text("# Rockets\n\nRockets launch into space.\n\n## Engines\n\nRocket engines burn fuel.")

    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    doc_id = resp.json()["doc_id"]

    resp = await http_client.post(
        "/search-grouped",
        json={"query": "rockets engines", "top_k": 5, "toc_only": False},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1

    # Grouped: doc-level fields at top, chunks nested
    r = results[0]
    assert r["doc_id"] == doc_id
    assert "brief_intro" in r
    assert "chunks" in r
    assert isinstance(r["chunks"], list)


async def test_search_grouped_toc_only(http_client, tmp_path):
    p = tmp_path / "toc.md"
    p.write_text("# A\n\nIntro.\n\n## B\n\nDetails.")

    await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})

    resp = await http_client.post(
        "/search-grouped",
        json={"query": "intro", "top_k": 5, "toc_only": True},
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) >= 1
    assert results[0]["chunks"] == []
    assert isinstance(results[0]["toc_entries"], list)


# ── Doc Meta ─────────────────────────────────────────────────────────


async def test_doc_meta_endpoint(http_client, tmp_path):
    p = tmp_path / "meta.md"
    p.write_text("# Meta\n\nContent.\n\n## Section\n\nMore.")

    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    doc_id = resp.json()["doc_id"]

    resp = await http_client.get(f"/doc-meta/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == doc_id
    assert "brief_intro" in data
    assert "instructions" in data
    assert "toc_entries" in data
    assert "source" in data
    assert "metadata" in data
    assert "processing_status" in data


async def test_doc_meta_404(http_client):
    resp = await http_client.get("/doc-meta/nonexistent-id")
    assert resp.status_code == 404


# ── Status ───────────────────────────────────────────────────────────


async def test_status_endpoint(http_client, tmp_path):
    p = tmp_path / "status.md"
    p.write_text("# Status\n\nContent.")

    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    doc_id = resp.json()["doc_id"]

    resp = await http_client.get(f"/status/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["doc_id"] == doc_id
    assert resp.json()["status"] == "completed"


async def test_status_not_found(http_client):
    resp = await http_client.get("/status/nonexistent-id")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


# ── Inspect ──────────────────────────────────────────────────────────


async def test_inspect_endpoint(http_client, tmp_path):
    p = tmp_path / "inspect.md"
    p.write_text("# Inspect\n\nOne.\n\n## Two\n\nMore")

    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    doc_id = resp.json()["doc_id"]

    resp = await http_client.get(f"/inspect/{doc_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == doc_id
    assert "brief_intro" in data
    assert "toc" in data


async def test_inspect_404(http_client):
    resp = await http_client.get("/inspect/nonexistent-id")
    assert resp.status_code == 404


# ── Read Segment ─────────────────────────────────────────────────────


async def test_read_segment_endpoint(http_client, tmp_path):
    p = tmp_path / "multi.md"
    p.write_text("# Chapter\n\nIntro.\n\n## Setup\n\nSetup steps.")

    resp = await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})
    doc_id = resp.json()["doc_id"]

    resp = await http_client.post(
        "/read-segment",
        json={"doc_id": doc_id, "section_path": "Setup"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["doc_id"] == doc_id
    assert "chunks" in data


async def test_read_segment_404(http_client):
    resp = await http_client.post(
        "/read-segment",
        json={"doc_id": "nonexistent-id", "section_path": "anything"},
    )
    assert resp.status_code == 404


# ── Index Content ────────────────────────────────────────────────────


async def test_index_content_endpoint(http_client):
    resp = await http_client.post(
        "/index-content",
        json={
            "content": "# Hello\n\nWorld of rockets.",
            "filename": "hello.md",
            "wait": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "doc_id" in data

    # Should be searchable
    resp = await http_client.post("/search", json={"query": "rockets", "top_k": 5})
    assert resp.status_code == 200
    assert len(resp.json()["results"]) >= 1


# ── Stats ────────────────────────────────────────────────────────────


async def test_stats_endpoint(http_client, tmp_path):
    p = tmp_path / "stats.md"
    p.write_text("# Stats\n\nContent.")

    await http_client.post("/index", json={"path": str(p), "force": True, "wait": True})

    resp = await http_client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "docs_indexed" in data
    assert data["docs_indexed"] >= 1


# ── Index file not found ─────────────────────────────────────────────


async def test_index_file_not_found(http_client):
    resp = await http_client.post("/index", json={"path": "/nonexistent/file.md"})
    assert resp.status_code == 404
