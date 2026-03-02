import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import litellm

from mentat.probes.base import Chunk, ProbeResult

logger = logging.getLogger("mentat.librarian")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CHUNK_SUMMARY_SYSTEM = """\
You are a concise technical summariser for the Mentat RAG system.
You will receive one or more content chunks from a file together with their
section/context information. For EACH chunk, produce a 1-3 sentence summary
that captures the key information.

Rules:
- Keep summaries factual and information-dense.
- Preserve important identifiers (names, numbers, column headers, function names).
- If a chunk is already very short (< 50 words), you may return it verbatim.
- Output valid JSON — an array of objects with "index" and "summary" keys.
"""


class Librarian:
    """Layer 3: Chunk summarisation and template-based instruction generation.

    Phase 1 — **Chunk Summarisation**: For each content chunk produced by the
    Probe layer, call a (cheap/fast) LLM to generate a brief summary.  Small
    files whose full content was already captured are skipped.

    Phase 2 — **Instruction Generation**: Template-based generation that uses
    ToC, statistics, and probe metadata to produce:
      * ``brief_intro`` — 1-2 sentence overview.
      * ``instructions`` — actionable reading guide that tells downstream
        models what data is present, what is *missing* (truncated/sampled),
        and how to access the raw file for details.
    """

    def __init__(
        self,
        summary_model: str = "gpt-4o-mini",
        summary_batch_size: int = 10,
        summary_api_key: Optional[str] = None,
        summary_api_base: Optional[str] = None,
    ):
        # Chunk summarisation (fast/cheap model)
        self.summary_model = summary_model
        self.summary_api_key = summary_api_key
        self.summary_api_base = summary_api_base
        self.summary_batch_size = summary_batch_size

    def _llm_kwargs(self) -> Dict[str, Any]:
        """Return kwargs for ``litellm.acompletion`` calls (summary model only)."""
        kw: Dict[str, Any] = {}
        if self.summary_api_key:
            kw["api_key"] = self.summary_api_key
        if self.summary_api_base:
            kw["api_base"] = self.summary_api_base
        return kw

    # ------------------------------------------------------------------
    # Phase 1: Per-chunk summarisation
    # ------------------------------------------------------------------

    # Chunks shorter than this (in estimated tokens) are kept verbatim —
    # an LLM summary would not meaningfully compress them.
    SHORT_CHUNK_TOKEN_THRESHOLD = 100

    async def summarize_chunks(
        self, probe_result: ProbeResult
    ) -> List[str]:
        """Return a list of summaries aligned 1-to-1 with ``probe_result.chunks``.

        Bypass rules (no LLM call):
          * Small files (``is_full_content``) — content IS the summary.
          * Individual short chunks (< SHORT_CHUNK_TOKEN_THRESHOLD tokens) —
            returned verbatim since a summary would not be shorter.
        """
        from mentat.probes._utils import estimate_tokens

        chunks = probe_result.chunks
        if not chunks:
            return []

        # Small-file bypass: content IS the summary
        if probe_result.stats.get("is_full_content"):
            logger.debug(f"Skipping summarization for {probe_result.filename} (full content)")
            return [c.content for c in chunks]

        # Separate short chunks (keep verbatim) from long ones (need LLM)
        summaries: List[Optional[str]] = [None] * len(chunks)
        long_chunks = []
        short_count = 0
        for i, chunk in enumerate(chunks):
            if estimate_tokens(chunk.content) < self.SHORT_CHUNK_TOKEN_THRESHOLD:
                summaries[i] = chunk.content
                short_count += 1
            else:
                long_chunks.append((i, chunk))

        if short_count:
            logger.debug(
                f"Skipping summarization for {short_count}/{len(chunks)} short chunks "
                f"(< {self.SHORT_CHUNK_TOKEN_THRESHOLD} tokens)"
            )

        # If all chunks are short, we're done
        if not long_chunks:
            return summaries  # type: ignore[return-value]

        # Batch the remaining long chunks
        only_chunks = [c for _, c in long_chunks]
        batches = [
            only_chunks[i : i + self.summary_batch_size]
            for i in range(0, len(only_chunks), self.summary_batch_size)
        ]

        logger.info(
            f"Summarizing {len(only_chunks)} chunks in {len(batches)} batches "
            f"({self.summary_batch_size} chunks/batch target) for {probe_result.filename}"
        )

        import time as time_module
        batch_start = time_module.perf_counter()
        tasks = [self._summarize_batch(batch, probe_result) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        batch_duration = time_module.perf_counter() - batch_start

        avg_per_batch = batch_duration / len(batches) if batches else 0
        logger.info(
            f"Completed summarization for {probe_result.filename} "
            f"({batch_duration:.1f}s total, {avg_per_batch:.1f}s avg/batch)"
        )

        # Map batch results back to original indices via long_chunks
        long_offset = 0
        for batch, result in zip(batches, batch_results):
            if isinstance(result, Exception):
                logger.warning("Chunk summary batch failed: %s", result)
                for j, _chunk in enumerate(batch):
                    orig_idx = long_chunks[long_offset + j][0]
                    summaries[orig_idx] = _chunk.content[:200]
            else:
                for j, summary in enumerate(result):
                    orig_idx = long_chunks[long_offset + j][0]
                    summaries[orig_idx] = summary
            long_offset += len(batch)

        # Safety: fill any remaining None slots
        for i, s in enumerate(summaries):
            if s is None:
                summaries[i] = chunks[i].content[:200]

        return summaries  # type: ignore[return-value]

    async def _summarize_batch(
        self, chunks: List[Chunk], probe_result: ProbeResult
    ) -> List[str]:
        """Summarise a batch of chunks in a single LLM call."""
        import time as time_module
        batch_id = chunks[0].index if chunks else 0
        batch_length = sum(len(c.content) for c in chunks)

        logger.debug(
            f"Batch {batch_id} starting ({len(chunks)} chunks, {batch_length} chars)..."
        )
        batch_start = time_module.perf_counter()

        entries = []
        for c in chunks:
            entry: Dict[str, Any] = {"index": c.index, "content": c.content}
            if c.section:
                entry["section"] = c.section
            entries.append(entry)

        user_msg = (
            f"File: {probe_result.filename} ({probe_result.file_type})\n\n"
            f"Chunks:\n{json.dumps(entries, ensure_ascii=False, default=str)}\n\n"
            f"Return a JSON array of objects: "
            f'[{{"index": <int>, "summary": "<string>"}}, ...]'
        )

        response = await litellm.acompletion(
            model=self.summary_model,
            messages=[
                {"role": "system", "content": _CHUNK_SUMMARY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            **self._llm_kwargs(),
        )

        batch_duration = time_module.perf_counter() - batch_start
        logger.debug(f"Batch {batch_id} completed in {batch_duration:.1f}s")

        content = response.choices[0].message.content
        data = json.loads(content)

        # Handle both {"summaries": [...]} and bare [...]
        if isinstance(data, dict):
            data = data.get("summaries", data.get("chunks", list(data.values())[0]))
        if not isinstance(data, list):
            data = [data]

        # Build index → summary map and align with input order
        summary_map: Dict[int, str] = {}
        for item in data:
            if isinstance(item, dict):
                summary_map[item.get("index", -1)] = item.get("summary", "")
            elif isinstance(item, str):
                # Fallback: positional
                pass

        result = []
        for i, chunk in enumerate(chunks):
            s = summary_map.get(chunk.index, "")
            if not s and i < len(data) and isinstance(data[i], dict):
                s = data[i].get("summary", "")
            if not s:
                s = chunk.content[:200]
            result.append(s)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_brief_intro(brief_intro: str, probe_result: ProbeResult) -> str:
        """Prepend the document's topic title/abstract to brief_intro.

        Probe-generated brief_intro describes the extraction *method* (e.g.
        "Markdown document analyzed via regex").  For discovery (toc_only
        search, get_summary), the consumer needs to know what the document
        is *about*.  This method prepends a content-aware prefix.
        """
        topic = probe_result.topic
        parts = []

        # Add document title
        if topic.title:
            parts.append(topic.title)

        # Add first paragraph / abstract if available and different from title
        if topic.abstract:
            parts.append(topic.abstract)
        elif topic.first_paragraph:
            # Truncate to keep it brief
            fp = topic.first_paragraph
            if len(fp) > 150:
                fp = fp[:147] + "..."
            parts.append(fp)

        if parts:
            content_summary = " — ".join(parts)
            return f"{content_summary}\n\n{brief_intro}"

        return brief_intro

    # ------------------------------------------------------------------
    # Phase 2: Template-based instruction generation (no LLM)
    # ------------------------------------------------------------------

    def generate_guide_template(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate instructions via template (no LLM call).

        Returns ``(brief_intro, instructions)`` based purely on probe metadata.
        This is ~10x faster than LLM-based generation.

        If the probe already generated format-specific instructions, use those
        instead of the generic fallback templates.
        """
        # Check for probe-generated instructions first
        if probe_result.brief_intro and probe_result.instructions:
            logger.debug(f"Using probe-generated instructions for {probe_result.filename}")
            brief_intro = probe_result.brief_intro
            # Enrich brief_intro with the document's actual topic/title
            # so it's useful for discovery, not just a probe method description
            brief_intro = self._enrich_brief_intro(brief_intro, probe_result)
            return brief_intro, probe_result.instructions

        # Fallback to generic template generation
        logger.debug(f"Using fallback template generation for {probe_result.filename}")
        stats = probe_result.stats
        structure = probe_result.structure
        topic = probe_result.topic

        # Build brief intro
        chunks_count = len(probe_result.chunks)
        total_tokens = stats.get("total_tokens", stats.get("approx_tokens", 0))
        file_type = probe_result.file_type.upper()

        brief_intro = f"This {file_type} file"
        if topic.title:
            brief_intro += f" titled '{topic.title}'"
        brief_intro += f" contains {chunks_count} section{'s' if chunks_count != 1 else ''}"
        if total_tokens:
            brief_intro += f" totaling ~{total_tokens:,} tokens"
        brief_intro += "."

        # Build instructions
        parts = ["Available information:"]

        # Structure summary
        if structure.toc:
            toc_count = len(structure.toc)
            parts.append(f"- Structure with {toc_count} sections")
            # List top-level sections
            top_sections = [e.title for e in structure.toc if e.level == 1][:5]
            if top_sections:
                parts.append(f"  ({', '.join(top_sections)}{'...' if toc_count > 5 else ''})")

        if structure.columns:
            parts.append(f"- Columns: {', '.join(structure.columns[:10])}")

        if structure.schema_tree:
            parts.append(f"- Schema information for data structure")

        # Truncation/sampling notes
        truncation_notes = []
        if stats.get("is_truncated"):
            truncation_notes.append("some sections are truncated")

        original_size = stats.get("original_size_tokens", 0)
        if original_size > total_tokens:
            pct_saved = 100 * (1 - total_tokens / max(1, original_size))
            truncation_notes.append(
                f"{pct_saved:.0f}% of content is summarized/sampled"
            )

        # File type specific notes
        if probe_result.file_type == "csv" and stats.get("total_rows"):
            sampled = stats.get("sample_rows", 0)
            total = stats.get("total_rows", 0)
            if sampled < total:
                truncation_notes.append(
                    f"only {sampled} of {total:,} rows sampled"
                )

        if probe_result.file_type == "code":
            truncation_notes.append("function bodies may be omitted (signatures shown)")

        if truncation_notes:
            parts.append("\nLimitations:")
            for note in truncation_notes:
                parts.append(f"- {note}")

        # Access suggestions
        parts.append(
            "\nFor complete details, access the original file using file read operations."
        )

        if stats.get("is_full_content"):
            parts[-1] = "Complete content is available in the chunks."

        instructions = "\n".join(parts)

        return brief_intro, instructions
