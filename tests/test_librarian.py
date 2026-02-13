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
        assert lib.model == "gpt-4o"
        assert lib.summary_model == "gpt-4o"
        assert lib.summary_batch_size == 10
        assert lib.api_key is None
        assert lib.api_base is None
        assert lib.summary_api_key is None
        assert lib.summary_api_base is None

    def test_custom_params(self):
        lib = Librarian(
            model="anthropic/claude-sonnet-4-5-20250929",
            summary_model="gpt-4o-mini",
            summary_batch_size=5,
            api_key="sk-inst",
            api_base="http://inst:8080",
            summary_api_key="sk-sum",
            summary_api_base="http://sum:9090",
        )
        assert lib.model == "anthropic/claude-sonnet-4-5-20250929"
        assert lib.summary_model == "gpt-4o-mini"
        assert lib.summary_batch_size == 5
        assert lib.api_key == "sk-inst"
        assert lib.api_base == "http://inst:8080"
        assert lib.summary_api_key == "sk-sum"
        assert lib.summary_api_base == "http://sum:9090"

    def test_summary_model_defaults_to_model(self):
        lib = Librarian(model="custom-model")
        assert lib.summary_model == "custom-model"

    def test_summary_credentials_fallback_to_instruction(self):
        lib = Librarian(api_key="sk-inst", api_base="http://inst:8080")
        assert lib.summary_api_key == "sk-inst"
        assert lib.summary_api_base == "http://inst:8080"

    def test_llm_kwargs_empty_when_no_credentials(self):
        lib = Librarian()
        assert lib._llm_kwargs("instruction") == {}
        assert lib._llm_kwargs("summary") == {}

    def test_llm_kwargs_instruction_credentials(self):
        lib = Librarian(
            api_key="sk-inst", api_base="http://inst:8080",
            summary_api_key="sk-sum", summary_api_base="http://sum:9090",
        )
        kw = lib._llm_kwargs("instruction")
        assert kw == {"api_key": "sk-inst", "api_base": "http://inst:8080"}

    def test_llm_kwargs_summary_credentials(self):
        lib = Librarian(
            api_key="sk-inst", api_base="http://inst:8080",
            summary_api_key="sk-sum", summary_api_base="http://sum:9090",
        )
        kw = lib._llm_kwargs("summary")
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
            api_key="sk-inst", api_base="http://inst:8080",
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


# ── Phase 2: Instruction Generation ─────────────────────────────────────────


class TestGenerateGuide:
    @pytest.mark.asyncio
    async def test_returns_brief_intro_and_instructions(self):
        lib = Librarian()
        pr = _make_probe_result()
        summaries = ["Sum 0", "Sum 1", "Sum 2"]

        mock_response = _mock_completion_response(
            {
                "brief_intro": "A test report about topics.",
                "instructions": "See Section 0 for details.",
            },
            tokens=150,
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            intro, instructions, tokens = await lib.generate_guide(
                pr, chunk_summaries=summaries
            )

        assert intro == "A test report about topics."
        assert instructions == "See Section 0 for details."
        assert tokens == 150

    @pytest.mark.asyncio
    async def test_works_without_summaries(self):
        """Backwards compatible: generate_guide works without chunk_summaries."""
        lib = Librarian()
        pr = _make_probe_result()

        mock_response = _mock_completion_response(
            {"brief_intro": "Intro", "instructions": "Instructions"},
            tokens=100,
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            intro, instructions, tokens = await lib.generate_guide(pr)

        assert intro == "Intro"
        assert instructions == "Instructions"

    @pytest.mark.asyncio
    async def test_forwards_instruction_credentials(self):
        lib = Librarian(
            api_key="sk-inst", api_base="http://inst:8080",
            summary_api_key="sk-sum", summary_api_base="http://sum:9090",
        )
        pr = _make_probe_result()

        mock_response = _mock_completion_response(
            {"brief_intro": "X", "instructions": "Y"}
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            await lib.generate_guide(pr)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-inst"
        assert call_kwargs.kwargs["api_base"] == "http://inst:8080"

    @pytest.mark.asyncio
    async def test_uses_instruction_model_not_summary_model(self):
        lib = Librarian(model="expensive-model", summary_model="cheap-model")
        pr = _make_probe_result()

        mock_response = _mock_completion_response(
            {"brief_intro": "X", "instructions": "Y"}
        )

        with patch("mentat.librarian.engine.litellm") as mock_litellm:
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            await lib.generate_guide(pr)

        call_kwargs = mock_litellm.acompletion.call_args
        assert call_kwargs.kwargs["model"] == "expensive-model"


# ── Prompt Building ──────────────────────────────────────────────────────────


class TestBuildInstructionPrompt:
    def test_includes_file_info(self):
        lib = Librarian()
        pr = _make_probe_result(filename="data.csv", file_type="csv")
        prompt = lib._build_instruction_prompt(pr)
        assert "data.csv" in prompt
        assert "csv" in prompt

    def test_includes_topic(self):
        lib = Librarian()
        pr = _make_probe_result(title="My Report")
        prompt = lib._build_instruction_prompt(pr)
        assert "My Report" in prompt
        assert "An abstract for testing" in prompt

    def test_includes_toc(self):
        lib = Librarian()
        pr = _make_probe_result()
        prompt = lib._build_instruction_prompt(pr)
        assert "Table of Contents" in prompt
        assert "Section 0" in prompt

    def test_includes_chunk_summaries_when_provided(self):
        lib = Librarian()
        pr = _make_probe_result(num_chunks=2)
        summaries = ["Summary of first chunk", "Summary of second chunk"]
        prompt = lib._build_instruction_prompt(pr, chunk_summaries=summaries)
        assert "[Chunk Summaries]" in prompt
        assert "Summary of first chunk" in prompt
        assert "Summary of second chunk" in prompt
        # Section context should be preserved
        assert "[Section 0]" in prompt

    def test_without_summaries_shows_chunk_count(self):
        lib = Librarian()
        pr = _make_probe_result(num_chunks=5)
        prompt = lib._build_instruction_prompt(pr)
        assert "5 chunks available" in prompt
        assert "[Chunk Summaries]" not in prompt

    def test_full_content_flag(self):
        lib = Librarian()
        pr = _make_probe_result(is_full_content=True)
        prompt = lib._build_instruction_prompt(pr)
        assert "Full content provided" in prompt

    def test_mentions_missing_data_guidance(self):
        lib = Librarian()
        pr = _make_probe_result()
        prompt = lib._build_instruction_prompt(pr, chunk_summaries=["S1", "S2", "S3"])
        assert "MISSING" in prompt
        assert "original file" in prompt.lower()

    def test_includes_columns_for_csv(self):
        lib = Librarian()
        pr = _make_probe_result(file_type="csv")
        pr.structure.columns = ["id", "name", "value"]
        prompt = lib._build_instruction_prompt(pr)
        assert "id, name, value" in prompt

    def test_includes_definitions_for_code(self):
        lib = Librarian()
        pr = _make_probe_result(file_type="code")
        pr.structure.definitions = ["class Parser", "def parse()", "def tokenize()"]
        prompt = lib._build_instruction_prompt(pr)
        assert "class Parser" in prompt
        assert "def parse()" in prompt
