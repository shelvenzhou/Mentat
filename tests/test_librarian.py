"""Tests for the Librarian layer (mentat/librarian/engine.py).

These tests use mocked LLM calls so they run without API keys.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mentat.librarian.engine import Librarian
from mentat.probes.base import (
    Chunk,
    ProbeResult,
    StructureInfo,
    TocEntry,
    TopicInfo,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_probe_result(
    filename="report.pdf",
    file_type="pdf",
    num_chunks=3,
    is_full_content=False,
    title="Test Report",
    toc_entries=None,
    stats_extra=None,
):
    """Helper to build a ProbeResult with sensible defaults."""
    chunks = [
        Chunk(
            content=f"Content of chunk {i} with some details about topic {i}.",
            index=i,
            section=f"Section {i}",
            page=i + 1,
        )
        for i in range(num_chunks)
    ]
    toc = toc_entries or [
        TocEntry(level=1, title=f"Section {i}", preview=f"Preview of section {i}")
        for i in range(num_chunks)
    ]
    stats = {"is_full_content": is_full_content, "total_pages": num_chunks}
    if stats_extra:
        stats.update(stats_extra)

    return ProbeResult(
        filename=filename,
        file_type=file_type,
        topic=TopicInfo(title=title, abstract="An abstract for testing."),
        structure=StructureInfo(toc=toc),
        stats=stats,
        chunks=chunks,
    )


def _mock_completion_response(data: dict, tokens: int = 100):
    """Build a mock litellm completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(data)
    resp.usage = MagicMock()
    resp.usage.total_tokens = tokens
    return resp


# ── Librarian init ───────────────────────────────────────────────────────────


class TestLibrarianInit:
    def test_defaults(self):
        lib = Librarian()
        assert lib.summary_model == "gpt-4o-mini"
        assert lib.summary_batch_size == 10
        assert lib.summary_api_key is None
        assert lib.summary_api_base is None

    def test_custom_params(self):
        lib = Librarian(
            summary_model="gpt-4o",
            summary_batch_size=5,
            summary_api_key="sk-sum",
            summary_api_base="http://sum:9090",
        )
        assert lib.summary_model == "gpt-4o"
        assert lib.summary_batch_size == 5
        assert lib.summary_api_key == "sk-sum"
        assert lib.summary_api_base == "http://sum:9090"

    def test_llm_kwargs_empty_when_no_credentials(self):
        lib = Librarian()
        assert lib._llm_kwargs() == {}

    def test_llm_kwargs_summary_credentials(self):
        lib = Librarian(
            summary_api_key="sk-sum", summary_api_base="http://sum:9090",
        )
        kw = lib._llm_kwargs()
        assert kw == {"api_key": "sk-sum", "api_base": "http://sum:9090"}


# ── Phase 1: Chunk Summarisation ─────────────────────────────────────────────


class TestSummarizeChunks:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        lib = Librarian()
        pr = _make_probe_result(num_chunks=0)
        result = await lib.summarize_chunks(pr)
        assert result == []

    @pytest.mark.asyncio
    async def test_small_file_bypass(self):
        """is_full_content files should return raw content as summary."""
        lib = Librarian()
        pr = _make_probe_result(num_chunks=2, is_full_content=True)
        result = await lib.summarize_chunks(pr)
        assert len(result) == 2
        assert result[0] == pr.chunks[0].content
        assert result[1] == pr.chunks[1].content

    @pytest.mark.asyncio
    async def test_calls_llm_for_large_files(self):
        lib = Librarian()
        pr = _make_probe_result(num_chunks=3, is_full_content=False)

        mock_response = _mock_completion_response(
            {
                "summaries": [
                    {"index": 0, "summary": "Summary of chunk 0"},
                    {"index": 1, "summary": "Summary of chunk 1"},
                    {"index": 2, "summary": "Summary of chunk 2"},
                ]
            }
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            result = await lib.summarize_chunks(pr)

        assert len(result) == 3
        assert result[0] == "Summary of chunk 0"
        assert result[1] == "Summary of chunk 1"
        assert result[2] == "Summary of chunk 2"
        mock_litellm.acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_batching(self):
        """Chunks exceeding batch_size should be split into multiple LLM calls."""
        lib = Librarian(summary_batch_size=2)
        pr = _make_probe_result(num_chunks=5, is_full_content=False)

        def make_response(batch_indices):
            return _mock_completion_response(
                {
                    "summaries": [
                        {"index": i, "summary": f"Summary {i}"}
                        for i in batch_indices
                    ]
                }
            )

        call_count = 0

        async def mock_acompletion(**kwargs):
            nonlocal call_count
            user_msg = kwargs["messages"][1]["content"]
            # Parse chunk indices from the user message
            chunks_data = json.loads(
                user_msg.split("Chunks:\n")[1].split("\n\nReturn")[0]
            )
            indices = [c["index"] for c in chunks_data]
            call_count += 1
            return make_response(indices)

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = mock_acompletion
            result = await lib.summarize_chunks(pr)

        assert len(result) == 5
        assert call_count == 3  # ceil(5/2) = 3 batches
        for i in range(5):
            assert result[i] == f"Summary {i}"

    @pytest.mark.asyncio
    async def test_fallback_on_llm_failure(self):
        """If LLM call fails, should fall back to content[:200]."""
        lib = Librarian()
        pr = _make_probe_result(num_chunks=2, is_full_content=False)

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(
                side_effect=RuntimeError("API error")
            )
            result = await lib.summarize_chunks(pr)

        assert len(result) == 2
        # Should fall back to truncated content
        assert result[0] == pr.chunks[0].content[:200]
        assert result[1] == pr.chunks[1].content[:200]

    @pytest.mark.asyncio
    async def test_forwards_summary_credentials(self):
        lib = Librarian(
            summary_api_key="sk-sum", summary_api_base="http://sum:9090",
        )
        pr = _make_probe_result(num_chunks=1, is_full_content=False)

        mock_response = _mock_completion_response(
            {"summaries": [{"index": 0, "summary": "Summary"}]}
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            await lib.summarize_chunks(pr)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-sum"
        assert call_kwargs.kwargs["api_base"] == "http://sum:9090"


# ── Phase 2: Template-based Instruction Generation ──────────────────────────


class TestGenerateGuideTemplate:
    def test_generates_brief_intro(self):
        lib = Librarian()
        pr = _make_probe_result(title="Test Report", num_chunks=3)
        pr.stats["total_tokens"] = 1500

        intro, instructions = lib.generate_guide_template(pr)

        assert "PDF" in intro
        assert "Test Report" in intro
        assert "3 sections" in intro
        assert "1,500 tokens" in intro

    def test_generates_instructions_with_structure(self):
        lib = Librarian()
        pr = _make_probe_result(num_chunks=3)

        intro, instructions = lib.generate_guide_template(pr)

        assert "Available information:" in instructions
        assert "Structure with 3 sections" in instructions
        assert "Section 0" in instructions

    def test_includes_truncation_notes(self):
        lib = Librarian()
        pr = _make_probe_result()
        pr.stats["is_truncated"] = True
        pr.stats["original_size_tokens"] = 5000
        pr.stats["total_tokens"] = 1000

        intro, instructions = lib.generate_guide_template(pr)

        assert "Limitations:" in instructions
        assert "truncated" in instructions.lower()

    def test_csv_specific_notes(self):
        lib = Librarian()
        pr = _make_probe_result(file_type="csv")
        pr.stats["total_rows"] = 10000
        pr.stats["sample_rows"] = 100
        pr.structure.columns = ["id", "name", "value"]

        intro, instructions = lib.generate_guide_template(pr)

        assert "Columns: id, name, value" in instructions
        assert "100 of 10,000 rows sampled" in instructions

    def test_full_content_flag(self):
        lib = Librarian()
        pr = _make_probe_result(is_full_content=True)

        intro, instructions = lib.generate_guide_template(pr)

        assert "Complete content is available" in instructions
