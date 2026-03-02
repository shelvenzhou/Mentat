"""Tests for get_doc_meta() — lightweight document metadata retrieval."""


async def test_doc_meta_basic(mentat_instance, tmp_path):
    """get_doc_meta returns all expected fields."""
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nSome content.\n\n## Section\n\nMore content.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    meta = await mentat_instance.get_doc_meta(doc_id)

    assert meta is not None
    assert meta["doc_id"] == doc_id
    assert meta["filename"] != ""
    assert "brief_intro" in meta
    assert "instructions" in meta
    assert "toc_entries" in meta
    assert "source" in meta
    assert "metadata" in meta
    assert "processing_status" in meta


async def test_doc_meta_not_found(mentat_instance):
    """Returns None for non-existent doc_id."""
    meta = await mentat_instance.get_doc_meta("nonexistent-id")
    assert meta is None


async def test_doc_meta_includes_source_and_metadata(mentat_instance, tmp_path):
    """Source and metadata fields are populated from what was passed to add()."""
    p = tmp_path / "page.md"
    p.write_text("# Page\n\nContent.")

    doc_id = await mentat_instance.add(
        str(p), force=True, wait=True,
        source="browser",
        metadata={"url": "https://example.com"},
    )

    meta = await mentat_instance.get_doc_meta(doc_id)
    assert meta is not None
    assert meta["source"] == "browser"
    assert meta["metadata"]["url"] == "https://example.com"


async def test_doc_meta_includes_processing_status(mentat_instance, tmp_path):
    """processing_status is present and reflects completed state after wait=True."""
    p = tmp_path / "status.md"
    p.write_text("# Status\n\nContent.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    meta = await mentat_instance.get_doc_meta(doc_id)

    assert meta is not None
    assert meta["processing_status"] == "completed"


async def test_doc_meta_includes_instructions(mentat_instance, tmp_path):
    """instructions field is populated (unlike inspect lightweight mode)."""
    p = tmp_path / "instr.md"
    p.write_text("# Guide\n\nHow to do things.\n\n## Step 1\n\nFirst step.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    meta = await mentat_instance.get_doc_meta(doc_id)

    assert meta is not None
    # Template-based instructions should produce non-empty string
    assert meta["instructions"] is not None


async def test_doc_meta_includes_toc_entries(mentat_instance, tmp_path):
    """toc_entries extracted from probe data."""
    p = tmp_path / "toc.md"
    p.write_text(
        "# Main\n\nIntro.\n\n"
        "## Section A\n\nContent A.\n\n"
        "## Section B\n\nContent B."
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    meta = await mentat_instance.get_doc_meta(doc_id)

    assert meta is not None
    assert isinstance(meta["toc_entries"], list)
    assert len(meta["toc_entries"]) >= 1


async def test_doc_meta_default_source_empty(mentat_instance, tmp_path):
    """Without source, defaults to empty string."""
    p = tmp_path / "plain.md"
    p.write_text("# Plain\n\nContent.")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    meta = await mentat_instance.get_doc_meta(doc_id)

    assert meta is not None
    assert meta["source"] == ""
    assert meta["metadata"] == {}
