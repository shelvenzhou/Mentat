import pytest

from mentat.probes import run_probe
from mentat.probes.base import ProbeResult


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
async def test_smoke_add_and_search(mentat_instance, tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Orbit\n\nThe moon orbits the earth.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search("moon orbit", top_k=5)

    assert isinstance(doc_id, str)
    assert any(r.doc_id == doc_id for r in results)


@pytest.mark.asyncio
async def test_smoke_add_batch(mentat_instance, tmp_path):
    p1 = tmp_path / "a.md"
    p2 = tmp_path / "b.md"
    p1.write_text("# A\n\nalpha")
    p2.write_text("# B\n\nbeta")

    ids = await mentat_instance.add_batch([str(p1), str(p2)], force=True, summarize=False)

    assert len(ids) == 2
    assert all(isinstance(i, str) for i in ids)
    assert mentat_instance.storage.count_docs() == 2


@pytest.mark.asyncio
async def test_smoke_inspect(mentat_instance, tmp_path):
    p = tmp_path / "inspect.md"
    p.write_text("# Inspect\n\nOne.\n\n## Two\nMore")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    info = await mentat_instance.inspect(doc_id, full=True)

    assert info is not None
    assert info["doc_id"] == doc_id
    assert "probe" in info
    assert "chunk_summaries" in info


@pytest.mark.asyncio
async def test_smoke_collections(mentat_instance, tmp_path):
    p = tmp_path / "col.md"
    p.write_text("# Collection\n\nScoped search content")

    await mentat_instance.start()
    col = mentat_instance.collection("team-notes")

    doc_id = await col.add(str(p), force=True, summarize=False)
    done = await mentat_instance.wait_for_completion(doc_id, timeout=20)
    results = await col.search("scoped search", top_k=5)

    assert done is True
    assert doc_id in col.doc_ids
    assert len(results) >= 1
    assert all(r.doc_id in col.doc_ids for r in results)


@pytest.mark.asyncio
async def test_smoke_stats(mentat_instance, tmp_path):
    p = tmp_path / "stats.md"
    p.write_text("# Stats\n\ncontent")

    await mentat_instance.add(str(p), force=True, wait=True)
    s = mentat_instance.stats()

    assert "docs_indexed" in s
    assert "chunks_stored" in s
    assert "storage_size_bytes" in s
    assert "access_tracker" in s
    assert s["docs_indexed"] >= 1


# ── ToC-only Search Tests ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_toc_only(mentat_instance, tmp_path):
    p = tmp_path / "toc_doc.md"
    p.write_text(
        "# Chapter 1\n\nIntroduction text here.\n\n"
        "## Section A\n\nDetails about section A.\n\n"
        "## Section B\n\nDetails about section B."
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search("introduction", top_k=5, toc_only=True)

    assert len(results) >= 1
    r = results[0]
    # In toc_only mode, content should be empty
    assert r.content == ""
    assert r.brief_intro != ""
    assert r.doc_id == doc_id
    # toc_entries should be populated
    assert isinstance(r.toc_entries, list)
    assert len(r.toc_entries) >= 1


@pytest.mark.asyncio
async def test_search_toc_only_shows_matched_sections(mentat_instance, tmp_path):
    p = tmp_path / "multi.md"
    p.write_text(
        "# Main\n\nMain intro.\n\n"
        "## Installation\n\nInstall steps.\n\n"
        "## Usage\n\nUsage info."
    )

    await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search("install steps", top_k=5, toc_only=True)

    assert len(results) >= 1
    # toc_entries should contain the document's table of contents
    assert isinstance(results[0].toc_entries, list)
    assert len(results[0].toc_entries) >= 1


# ── read_segment Tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_read_segment(mentat_instance, tmp_path):
    p = tmp_path / "multi_section.md"
    p.write_text(
        "# Chapter 1\n\nIntro text.\n\n"
        "## Setup\n\nSetup instructions here.\n\n"
        "## Usage\n\nUsage details here."
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    result = await mentat_instance.read_segment(doc_id, "Setup")

    assert result["doc_id"] == doc_id
    assert result["filename"] != ""
    assert result["section_path"] == "Setup"
    assert isinstance(result["chunks"], list)
    assert isinstance(result["token_estimate"], int)


@pytest.mark.asyncio
async def test_read_segment_not_found(mentat_instance):
    result = await mentat_instance.read_segment("nonexistent-id", "anything")
    assert result.get("error") == "document_not_found"
    assert result["chunks"] == []


@pytest.mark.asyncio
async def test_read_segment_no_matching_section(mentat_instance, tmp_path):
    p = tmp_path / "simple.md"
    p.write_text("# Title\n\nSome content here.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    result = await mentat_instance.read_segment(doc_id, "NonExistentSection")

    assert result["doc_id"] == doc_id
    assert result["chunks"] == []
    assert "note" in result
