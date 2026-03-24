"""Tests for search_grouped() — document-grouped search results."""

from mentat.core.models import MentatDocResult, ChunkResult


async def test_search_grouped_basic(mentat_instance, tmp_path):
    """search_grouped returns MentatDocResult with nested chunks."""
    p = tmp_path / "doc.md"
    p.write_text("# Orbit\n\nThe moon orbits the earth.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped("moon orbit", top_k=5)

    assert len(results) >= 1
    r = results[0]
    assert isinstance(r, MentatDocResult)
    assert r.doc_id == doc_id
    assert r.filename != ""
    assert len(r.chunks) >= 1
    for chunk in r.chunks:
        assert isinstance(chunk, ChunkResult)
        assert chunk.chunk_id != ""


async def test_search_grouped_no_duplicate_metadata(mentat_instance, tmp_path):
    """Multiple chunks from same doc appear under one MentatDocResult."""
    p = tmp_path / "multi.md"
    p.write_text(
        "# Chapter 1\n\nIntroduction about the solar system.\n\n"
        "## Planets\n\nMercury Venus Earth Mars Jupiter Saturn.\n\n"
        "## Stars\n\nThe sun is a star in the milky way galaxy."
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped("solar system planets", top_k=10)

    # Should have exactly one doc result (not one per chunk)
    doc_results = [r for r in results if r.doc_id == doc_id]
    assert len(doc_results) == 1

    r = doc_results[0]
    assert r.filename != ""
    # All matching chunks should be nested
    assert len(r.chunks) >= 1


async def test_search_grouped_toc_only(mentat_instance, tmp_path):
    """In toc_only mode, chunks list is empty but toc_entries populated."""
    p = tmp_path / "toc_doc.md"
    p.write_text(
        "# Chapter 1\n\nIntro.\n\n"
        "## Section A\n\nDetails A.\n\n"
        "## Section B\n\nDetails B."
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped(
        "intro details", top_k=5, toc_only=True
    )

    assert len(results) >= 1
    r = results[0]
    assert r.doc_id == doc_id
    assert r.chunks == []  # toc_only => no chunks
    assert isinstance(r.toc_entries, list)
    assert len(r.toc_entries) >= 1
    assert r.brief_intro != ""


async def test_search_grouped_with_metadata_false(mentat_instance, tmp_path):
    """with_metadata=False leaves brief_intro/instructions/toc_entries empty."""
    p = tmp_path / "meta.md"
    p.write_text("# Test\n\nContent about rockets.")

    await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped(
        "rockets", top_k=5, with_metadata=False
    )

    assert len(results) >= 1
    r = results[0]
    assert r.brief_intro == ""
    assert r.toc_entries == []


async def test_search_grouped_with_metadata_true(mentat_instance, tmp_path):
    """with_metadata=True populates brief_intro and toc_entries."""
    p = tmp_path / "meta2.md"
    p.write_text("# Guide\n\nHow to build rockets.\n\n## Materials\n\nSteel and aluminum.")

    await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped(
        "rockets", top_k=5, with_metadata=True
    )

    assert len(results) >= 1
    r = results[0]
    assert r.brief_intro != ""
    assert isinstance(r.toc_entries, list)


async def test_search_grouped_source_filter(mentat_instance, tmp_path):
    """Source filter works in grouped mode."""
    p1 = tmp_path / "web.md"
    p1.write_text("# Web Page\n\nMoon facts.")
    p2 = tmp_path / "email.md"
    p2.write_text("# Email\n\nMoon newsletter.")

    d1 = await mentat_instance.add(str(p1), force=True, wait=True, source="web_fetch")
    d2 = await mentat_instance.add(str(p2), force=True, wait=True, source="composio:gmail")

    results = await mentat_instance.search_grouped(
        "moon", top_k=10, source="web_fetch"
    )
    doc_ids = {r.doc_id for r in results}
    assert d1 in doc_ids
    assert d2 not in doc_ids


async def test_search_grouped_best_score(mentat_instance, tmp_path):
    """Doc-level score should be the best (minimum) among its chunks."""
    p = tmp_path / "scored.md"
    p.write_text("# Score Test\n\nContent for scoring.")

    await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search_grouped("score test", top_k=5)

    assert len(results) >= 1
    r = results[0]
    if r.chunks:
        chunk_scores = [c.score for c in r.chunks]
        assert r.score == min(chunk_scores)


async def test_search_grouped_empty_results(mentat_instance):
    """Searching with no indexed docs returns empty list."""
    results = await mentat_instance.search_grouped("nothing here", top_k=5)
    assert results == []


async def test_search_grouped_source_no_match(mentat_instance, tmp_path):
    """When source filter matches nothing, return empty list."""
    p = tmp_path / "doc.md"
    p.write_text("# Doc\n\nSome content.")
    await mentat_instance.add(str(p), force=True, wait=True, source="web_fetch")

    results = await mentat_instance.search_grouped(
        "content", top_k=5, source="nonexistent"
    )
    assert results == []
