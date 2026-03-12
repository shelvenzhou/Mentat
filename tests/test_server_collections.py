"""HTTP endpoint tests for the new collection CRUD and multi-collection search."""

import pytest
from httpx import AsyncClient, ASGITransport

from mentat.server import create_app
from mentat.core.hub import Mentat, MentatConfig
from tests.conftest import FakeStorage, FakeEmbedding


@pytest.fixture
async def http_client(tmp_path, monkeypatch):
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


# ── Collection CRUD ─────────────────────────────────────────────────────


async def test_create_collection(http_client):
    resp = await http_client.post(
        "/collections/memory",
        json={
            "metadata": {"type": "system"},
            "auto_add_sources": ["openclaw:*"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "memory"
    assert data["metadata"] == {"type": "system"}
    assert data["auto_add_sources"] == ["openclaw:*"]
    assert data["doc_count"] == 0


async def test_get_collection(http_client):
    await http_client.post("/collections/memory", json={"metadata": {"type": "system"}})

    resp = await http_client.get("/collections/memory")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "memory"
    assert data["metadata"] == {"type": "system"}


async def test_get_collection_not_found(http_client):
    resp = await http_client.get("/collections/nope")
    assert resp.status_code == 404


async def test_update_collection(http_client):
    await http_client.post("/collections/memory", json={"metadata": {"v": 1}})

    resp = await http_client.put(
        "/collections/memory",
        json={"metadata": {"v": 2}},
    )
    assert resp.status_code == 200
    assert resp.json()["metadata"] == {"v": 2}


async def test_update_collection_not_found(http_client):
    resp = await http_client.put("/collections/nope", json={"metadata": {"v": 1}})
    assert resp.status_code == 404


async def test_delete_collection(http_client):
    await http_client.post("/collections/memory", json={})

    resp = await http_client.delete("/collections/memory")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == "memory"

    resp = await http_client.get("/collections/memory")
    assert resp.status_code == 404


async def test_delete_collection_not_found(http_client):
    resp = await http_client.delete("/collections/nope")
    assert resp.status_code == 404


async def test_list_collections(http_client):
    await http_client.post("/collections/memory", json={"metadata": {"type": "system"}})
    await http_client.post("/collections/files", json={"metadata": {"type": "system"}})

    resp = await http_client.get("/collections")
    assert resp.status_code == 200
    names = {c["name"] for c in resp.json()["collections"]}
    assert names == {"memory", "files"}


# ── GC ──────────────────────────────────────────────────────────────────


async def test_collections_gc(http_client):
    resp = await http_client.post("/collections/gc")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == []


# ── Index with collection param ─────────────────────────────────────────


async def test_index_with_collection(http_client, tmp_path):
    # Create collection
    await http_client.post("/collections/ses_abc", json={})

    # Index file into collection
    p = tmp_path / "doc.md"
    p.write_text("# Doc\nContent about auth.")

    resp = await http_client.post(
        "/index",
        json={"path": str(p), "force": True, "wait": True, "collection": "ses_abc"},
    )
    assert resp.status_code == 200
    doc_id = resp.json()["doc_id"]

    # Verify doc is in collection
    resp = await http_client.get("/collections/ses_abc")
    assert doc_id in resp.json()["doc_ids"]


async def test_index_content_with_collection(http_client):
    await http_client.post("/collections/ses_abc", json={})

    resp = await http_client.post(
        "/index-content",
        json={
            "content": "# Note\nAuth notes.",
            "filename": "note.md",
            "wait": True,
            "collection": "ses_abc",
        },
    )
    assert resp.status_code == 200
    doc_id = resp.json()["doc_id"]

    resp = await http_client.get("/collections/ses_abc")
    assert doc_id in resp.json()["doc_ids"]


# ── Search with collections param ──────────────────────────────────────


async def test_search_with_collections_param(http_client, tmp_path):
    # Create two collections
    await http_client.post("/collections/code", json={})
    await http_client.post("/collections/memory", json={})

    # Index into different collections
    p1 = tmp_path / "code.md"
    p1.write_text("# Code\nAuthentication middleware.")
    resp1 = await http_client.post(
        "/index", json={"path": str(p1), "force": True, "wait": True, "collection": "code"}
    )
    id1 = resp1.json()["doc_id"]

    p2 = tmp_path / "note.md"
    p2.write_text("# Note\nAuthentication notes.")
    resp2 = await http_client.post(
        "/index", json={"path": str(p2), "force": True, "wait": True, "collection": "memory"}
    )
    id2 = resp2.json()["doc_id"]

    # Search only code collection
    resp = await http_client.post(
        "/search",
        json={"query": "authentication", "collections": ["code"]},
    )
    doc_ids = {r["doc_id"] for r in resp.json()["results"]}
    assert id1 in doc_ids
    assert id2 not in doc_ids

    # Search both collections
    resp = await http_client.post(
        "/search",
        json={"query": "authentication", "collections": ["code", "memory"]},
    )
    doc_ids = {r["doc_id"] for r in resp.json()["results"]}
    assert id1 in doc_ids
    assert id2 in doc_ids


async def test_search_with_single_collection_compat(http_client, tmp_path):
    """The old `collection` param still works."""
    await http_client.post("/collections/code", json={})

    p = tmp_path / "doc.md"
    p.write_text("# Doc\nSome content.")
    resp = await http_client.post(
        "/index", json={"path": str(p), "force": True, "wait": True, "collection": "code"}
    )
    doc_id = resp.json()["doc_id"]

    resp = await http_client.post(
        "/search",
        json={"query": "content", "collection": "code"},
    )
    doc_ids = {r["doc_id"] for r in resp.json()["results"]}
    assert doc_id in doc_ids


# ── Auto-routing via server ─────────────────────────────────────────────


async def test_auto_routing_via_index(http_client, tmp_path):
    # Create collection with auto_add_sources
    await http_client.post(
        "/collections/files",
        json={"auto_add_sources": ["openclaw:*"]},
    )

    p = tmp_path / "doc.md"
    p.write_text("# Doc\nContent.")

    resp = await http_client.post(
        "/index",
        json={"path": str(p), "force": True, "wait": True, "source": "openclaw:Read"},
    )
    doc_id = resp.json()["doc_id"]

    # Doc should be auto-routed to "files"
    resp = await http_client.get("/collections/files")
    assert doc_id in resp.json()["doc_ids"]
