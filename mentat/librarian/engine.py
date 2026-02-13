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

_INSTRUCTION_SYSTEM = """\
You are 'The Librarian' for the Mentat system.
Your job is to produce a concise Reading Guide for a downstream LLM that will
use this file.  The downstream model will see ONLY your guide (not the raw
file), so it must be useful on its own while staying compact.

Critical context — the data provided to you is a *semantic fingerprint*, not
the full file.  Chunks may have been truncated or sampled (e.g. only the first
rows of a CSV, only function signatures of code, only heading + preview for
long documents).  The original file is stored separately and can be accessed
if the downstream model needs more detail.
"""


class Librarian:
    """Layer 3: Two-phase instruction generation.

    Phase 1 — **Chunk Summarisation**: For each content chunk produced by the
    Probe layer, call a (cheap/fast) LLM to generate a brief summary.  Small
    files whose full content was already captured are skipped.

    Phase 2 — **Instruction Generation**: Feed the ToC, chunk summaries, and
    statistics to the LLM to produce:
      * ``brief_intro`` — 1-2 sentence overview.
      * ``instructions`` — actionable reading guide that tells downstream
        models what data is present, what is *missing* (truncated/sampled),
        and how to access the raw file for details.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        summary_model: Optional[str] = None,
        summary_batch_size: int = 10,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        summary_api_key: Optional[str] = None,
        summary_api_base: Optional[str] = None,
    ):
        # Phase 2 — instruction generation (smart model)
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        # Phase 1 — chunk summarisation (fast/cheap model)
        self.summary_model = summary_model or model
        self.summary_api_key = summary_api_key or api_key
        self.summary_api_base = summary_api_base or api_base
        self.summary_batch_size = summary_batch_size

    def _llm_kwargs(self, phase: str = "instruction") -> Dict[str, Any]:
        """Return kwargs for ``litellm.acompletion`` calls.

        *phase* is ``"summary"`` (Phase 1) or ``"instruction"`` (Phase 2).
        """
        if phase == "summary":
            key, base = self.summary_api_key, self.summary_api_base
        else:
            key, base = self.api_key, self.api_base
        kw: Dict[str, Any] = {}
        if key:
            kw["api_key"] = key
        if base:
            kw["api_base"] = base
        return kw

    # ------------------------------------------------------------------
    # Phase 1: Per-chunk summarisation
    # ------------------------------------------------------------------

    async def summarize_chunks(
        self, probe_result: ProbeResult
    ) -> List[str]:
        """Return a list of summaries aligned 1-to-1 with ``probe_result.chunks``.

        For small files (``is_full_content``), the raw content is returned as-is
        since it is already compact enough to serve as its own summary.
        """
        chunks = probe_result.chunks
        if not chunks:
            return []

        # Small-file bypass: content IS the summary
        if probe_result.stats.get("is_full_content"):
            return [c.content for c in chunks]

        # Batch chunks and call LLM
        summaries: List[Optional[str]] = [None] * len(chunks)
        batches = [
            chunks[i : i + self.summary_batch_size]
            for i in range(0, len(chunks), self.summary_batch_size)
        ]

        tasks = [self._summarize_batch(batch, probe_result) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        offset = 0
        for batch, result in zip(batches, batch_results):
            if isinstance(result, Exception):
                logger.warning("Chunk summary batch failed: %s", result)
                # Fallback: use first 200 chars of raw content
                for i, chunk in enumerate(batch):
                    summaries[offset + i] = chunk.content[:200]
            else:
                for i, summary in enumerate(result):
                    summaries[offset + i] = summary
            offset += len(batch)

        # Safety: fill any remaining None slots
        for i, s in enumerate(summaries):
            if s is None:
                summaries[i] = chunks[i].content[:200]

        return summaries  # type: ignore[return-value]

    async def _summarize_batch(
        self, chunks: List[Chunk], probe_result: ProbeResult
    ) -> List[str]:
        """Summarise a batch of chunks in a single LLM call."""
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
            **self._llm_kwargs("summary"),
        )

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
    # Phase 2: Document-level instruction generation
    # ------------------------------------------------------------------

    async def generate_guide(
        self,
        probe_result: ProbeResult,
        chunk_summaries: Optional[List[str]] = None,
    ) -> Tuple[str, str, int]:
        """Generate a reading guide from probe results + chunk summaries.

        Returns ``(brief_intro, instructions, total_tokens)``.

        If *chunk_summaries* is ``None``, falls back to the old prompt that
        only uses ToC / stats (backwards compatible).
        """
        prompt = self._build_instruction_prompt(probe_result, chunk_summaries)

        response = await litellm.acompletion(
            model=self.model,
            messages=[
                {"role": "system", "content": _INSTRUCTION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            **self._llm_kwargs("instruction"),
        )

        content = response.choices[0].message.content
        data = json.loads(content)
        token_usage = response.usage.total_tokens

        return data.get("brief_intro", ""), data.get("instructions", ""), token_usage

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def _build_instruction_prompt(
        self,
        probe_result: ProbeResult,
        chunk_summaries: Optional[List[str]] = None,
    ) -> str:
        sections: List[str] = []

        # -- File info ------------------------------------------------
        sections.append(
            f"[File Info]\nFilename: {probe_result.filename}\n"
            f"Type: {probe_result.file_type}"
        )

        # -- Topic ----------------------------------------------------
        topic_parts = []
        if probe_result.topic.title:
            topic_parts.append(f"Title: {probe_result.topic.title}")
        if probe_result.topic.abstract:
            topic_parts.append(f"Abstract: {probe_result.topic.abstract}")
        if probe_result.topic.first_paragraph:
            topic_parts.append(
                f"First paragraph: {probe_result.topic.first_paragraph[:300]}"
            )
        sections.append(
            "[Topic]\n"
            + ("\n".join(topic_parts) if topic_parts else "No topic info available.")
        )

        # -- Structure (ToC with previews/annotations) -----------------
        structure = probe_result.structure
        struct_parts = []
        if structure.toc:
            toc_lines = []
            for e in structure.toc[:30]:
                line = f"{'  ' * (e.level - 1)}- {e.title}"
                if e.annotation:
                    line += f"  ({e.annotation})"
                if e.page is not None:
                    line += f"  [p.{e.page}]"
                if e.preview:
                    line += f" — {e.preview}"
                toc_lines.append(line)
            struct_parts.append("Table of Contents:\n" + "\n".join(toc_lines))
        if structure.captions:
            caps = ", ".join(c.text[:60] for c in structure.captions[:10])
            struct_parts.append(f"Captions: {caps}")
        if structure.columns:
            struct_parts.append(f"Columns: {', '.join(structure.columns)}")
        if structure.schema_tree:
            struct_parts.append(
                f"Schema: {json.dumps(structure.schema_tree, default=str)[:500]}"
            )
        if structure.definitions:
            struct_parts.append(
                f"Definitions: {', '.join(structure.definitions[:20])}"
            )
        sections.append(
            "[Structure]\n"
            + ("\n".join(struct_parts) if struct_parts else "No structure info.")
        )

        # -- Statistics ------------------------------------------------
        stats_text = json.dumps(probe_result.stats, indent=2, default=str)[:600]
        sections.append(f"[Statistics]\n{stats_text}")

        # -- Chunk summaries (mapped to sections) ----------------------
        is_full = probe_result.stats.get("is_full_content", False)
        chunks = probe_result.chunks

        if chunk_summaries and chunks:
            summary_lines = []
            for chunk, summary in zip(chunks, chunk_summaries):
                sec = f"[{chunk.section}] " if chunk.section else ""
                pg = f" (p.{chunk.page})" if chunk.page else ""
                summary_lines.append(f"  Chunk {chunk.index}{pg}: {sec}{summary}")
            sections.append(
                "[Chunk Summaries]\n" + "\n".join(summary_lines)
            )
        else:
            sections.append(
                f"[Chunks]\n{len(chunks)} chunks available."
                + (" (Full content provided.)" if is_full else "")
            )

        # -- Task for the LLM -----------------------------------------
        sections.append(
            "[Your Task]\n"
            "Based on the above, produce a JSON object:\n"
            "{\n"
            '  "brief_intro": "1-2 sentence summary of what this file contains.",\n'
            '  "instructions": "Strategic reading guide (see rules below)."\n'
            "}\n\n"
            "Rules for 'instructions':\n"
            "- State what information IS available in the chunk summaries.\n"
            "- Explicitly call out what data is MISSING or truncated "
            "(e.g. 'Only the first 10 rows of 50,000 are sampled', "
            "'Function bodies are omitted — only signatures shown').\n"
            "- Suggest concrete actions to get the missing data: "
            "'read the original file via read_range()', "
            "'filter with pandas', 'run grep on the raw file', etc.\n"
            "- Reference specific ToC sections or chunk indices when useful.\n"
            "- Keep it concise — ideally under 200 words."
        )

        return "\n\n".join(sections)
