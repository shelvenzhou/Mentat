import pytest
from pydantic import ValidationError

from mentat.probes.base import Chunk, TocEntry, ProbeResult


def test_chunk_creation():
    chunk = Chunk(content="hello", index=1, section="Intro")
    assert chunk.content == "hello"
    assert chunk.index == 1
    assert chunk.section == "Intro"
    assert chunk.page is None


def test_toc_entry_defaults():
    toc = TocEntry()
    assert toc.level == 1
    assert toc.title == ""
    assert toc.page is None
    assert toc.preview is None
    assert toc.annotation is None


def test_probe_result_required_fields():
    with pytest.raises(ValidationError):
        ProbeResult(file_type="markdown")

    with pytest.raises(ValidationError):
        ProbeResult(filename="a.md")


def test_probe_result_serialization():
    obj = ProbeResult(
        filename="a.md",
        file_type="markdown",
        chunks=[Chunk(content="text", index=0)],
    )

    payload = obj.model_dump_json()
    parsed = ProbeResult.model_validate_json(payload)

    assert parsed.filename == "a.md"
    assert parsed.file_type == "markdown"
    assert len(parsed.chunks) == 1
    assert parsed.chunks[0].content == "text"
