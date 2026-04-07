"""Regression tests for the Phase 2 agent-owned wiki design."""

import json

import pytest
from httpx import ASGITransport, AsyncClient

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig
from mentat.server import create_app
from mentat.wiki.generator import WikiGenerator
from mentat.wiki.log import WikiLog
from tests.conftest import FakeEmbedding, FakeStorage


class _FakeCollectionsStore:
    def get(self, name):
        return None


class _FakeFileStore:
    pass


def _make_generator(tmp_path):
    storage = FakeStorage()
    storage.add_stub(
        "12345678-aaaa-bbbb-cccc-1234567890ab",
        "demo.md",
        "Demo brief",
        "",
        json.dumps(
            {
                "file_type": "markdown",
                "topic": {"title": "Demo Doc"},
                "structure": {"toc": [{"level": 1, "title": "Setup"}]},
                "chunks": [{"section": "Setup", "content": "Install dependencies."}],
            }
        ),
        source="test",
    )
    generator = WikiGenerator(
        wiki_dir=str(tmp_path / "wiki"),
        storage=storage,
        collections_store=_FakeCollectionsStore(),
        file_store=_FakeFileStore(),
    )
    return generator, storage


def test_wiki_log_append_event(tmp_path):
    log = WikiLog(tmp_path / "wiki")
    line = log.append_event("ingest", filename="demo.md", sid="12345678")

    text = log.path.read_text("utf-8")
    assert "## [" in line
    assert " ingest | demo.md | sid=12345678" in line
    assert line in text


def test_wiki_generator_resolve_url(tmp_path):
    generator, storage = _make_generator(tmp_path)
    stub = storage.get_stub("12345678-aaaa-bbbb-cccc-1234567890ab")
    generator.generate_page(stub)

    result = generator.resolve_url("/wiki/pages/12345678#setup")
    assert result["doc_id"] == "12345678-aaaa-bbbb-cccc-1234567890ab"
    assert result["section_path"] in {"Setup", "setup"}
    assert result["filename"] == "demo.md"


def test_section_pages_compact_multiline_briefs(tmp_path):
    generator, storage = _make_generator(tmp_path)
    storage.add_stub(
        "87654321-bbbb-cccc-dddd-abcdefabcdef",
        "notes.md",
        "Line one.\n\nLine two with | pipes | inside.",
        "",
        json.dumps({"file_type": "markdown", "chunks": []}),
        source="test",
    )
    generator._collections_store.get = lambda name: {
        "doc_ids": [
            "12345678-aaaa-bbbb-cccc-1234567890ab",
            "87654321-bbbb-cccc-dddd-abcdefabcdef",
        ]
    }

    page = generator.generate_memories_page()
    text = page.read_text("utf-8")

    assert "| [notes.md](/wiki/pages/87654321) | Line one. Line two with \\| pipes \\| inside. |" in text


@pytest.fixture
async def wiki_http_client(tmp_path, monkeypatch):
    monkeypatch.setattr("mentat.core.hub.LanceDBStorage", FakeStorage)
    cfg = MentatConfig(
        db_path=str(tmp_path / "db"),
        storage_dir=str(tmp_path / "files"),
        wiki_dir=str(tmp_path / "wiki"),
        max_concurrent_tasks=2,
    )
    Mentat.reset()
    m = Mentat.get_instance(cfg)
    m.embeddings = FakeEmbedding()

    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    (wiki_dir / "index.md").write_text("# Test Index\n\n- [Topic](/wiki/topics/topic-a)\n", "utf-8")
    (wiki_dir / "log.md").write_text("# Log\n", "utf-8")
    topics_dir = wiki_dir / "topics"
    topics_dir.mkdir(exist_ok=True)
    (topics_dir / "topic-a.md").write_text(
        (
            "# Topic A\n\n"
            "A supported claim.[^12345678]\n\n"
            "[^12345678]: [_append_demo: records[0:2]](/wiki/pages/12345678#records02)\n"
        ),
        "utf-8",
    )
    (topics_dir / "topic-a.verified.json").write_text(
        json.dumps(
            {
                "checked_at": "2026-04-07T02:00:00Z",
                "claims": [
                    {
                        "sentence": "A supported claim.",
                        "citation": "[^12345678]",
                        "verdict": "supported",
                        "evidence": "Demo evidence",
                    }
                ],
            }
        ),
        "utf-8",
    )

    app = create_app(cfg)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    await m.shutdown()
    Mentat.reset()


@pytest.mark.asyncio
async def test_wiki_phase2_routes_render_agent_owned_files(wiki_http_client):
    index_resp = await wiki_http_client.get("/wiki/")
    assert index_resp.status_code == 200
    assert "Test Index" in index_resp.text

    topics_resp = await wiki_http_client.get("/wiki/topics/")
    assert topics_resp.status_code == 200
    assert "Topic A" in topics_resp.text

    topic_resp = await wiki_http_client.get("/wiki/topics/topic-a")
    assert topic_resp.status_code == 200
    assert "verified:" in topic_resp.text
    assert "badge-supported" in topic_resp.text
    assert 'class="citation"' in topic_resp.text
    assert "Sources" in topic_resp.text
    assert "_append_demo: records[0:2]" in topic_resp.text
    assert '/wiki/pages/12345678#records02' in topic_resp.text
