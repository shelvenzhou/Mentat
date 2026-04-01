"""Probe for JSONL (JSON Lines / NDJSON) files.

Each line is parsed as an independent JSON object.  Supports configurable
filtering via :class:`RecordFilterSet`, text extraction via dot-paths,
label prefixes, and N-record grouping (e.g. turn-based chat chunking).

When no ``probe_config`` is provided, falls back to a generic mode where
every line becomes a chunk with the raw JSON as content.
"""

from __future__ import annotations

import json as json_module
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

from mentat.probes.base import (
    BaseProbe,
    Chunk,
    ProbeResult,
    StructureInfo,
    TocEntry,
    TopicInfo,
)
from mentat.probes._utils import estimate_tokens, safe_read_text, should_bypass
from mentat.probes.record_filter import RecordFilterSet, _MISSING, _resolve_dot_path


# ── Probe config ────────────────────────────────────────────────────────

class JSONLProbeConfig(BaseModel):
    """Configurable options for JSONL probing.

    All fields are optional; defaults produce a no-filter, raw-JSON-per-line
    chunking strategy.
    """

    filters: list[dict] = []
    """RecordFilter dicts, e.g. ``[{"field": "type", "op": "eq", "value": "message"}]``."""

    filter_logic: str = "AND"
    """How to combine filters: ``"AND"`` or ``"OR"``."""

    text_fields: list[str] = []
    """Dot-paths to extract text from each record.
    When set, chunk content is the joined extracted text instead of raw JSON.
    Supports list-of-dicts traversal (see :func:`record_filter._resolve_dot_path`).
    """

    label_field: Optional[str] = None
    """Dot-path for a label prefix on each record's text, e.g. ``"message.role"``
    produces chunks like ``"user: Hello world"``."""

    group_size: int = 1
    """Number of (filtered) records to merge into a single chunk.
    ``1`` = one chunk per record.  ``2`` = pairs (e.g. user+assistant turns).
    """

    group_separator: str = "\n"
    """Separator between grouped records within a chunk."""

    timestamp_field: Optional[str] = None
    """Dot-path to a timestamp field, used for ToC generation and chunk metadata."""

    text_strip_patterns: list[str] = []
    """Regex patterns to strip from extracted text (applied after extraction).
    Useful for removing injected metadata preambles.  Each pattern is applied
    with ``re.DOTALL`` so ``.`` matches newlines."""


def _parse_probe_config(raw: Any) -> Optional[JSONLProbeConfig]:
    """Parse probe_config from various input forms."""
    if raw is None:
        return None
    if isinstance(raw, JSONLProbeConfig):
        return raw
    if isinstance(raw, dict):
        return JSONLProbeConfig(**raw)
    return None


# ── Text extraction ─────────────────────────────────────────────────────

def _extract_text(
    record: dict,
    text_fields: list[str],
    strip_patterns: Optional[list[re.Pattern]] = None,
) -> str:
    """Extract and join text from a record using dot-path text_fields.

    For each field path, resolves the value.  If the resolved value is a
    list (from list-of-dicts traversal), joins non-empty string elements.
    Filters out non-string values silently.

    If *strip_patterns* is provided, each compiled regex is applied to the
    final joined text to remove matched substrings.
    """
    parts: list[str] = []
    for path in text_fields:
        val = _resolve_dot_path(record, path)
        if val is _MISSING:
            continue
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    parts.append(item)
        else:
            parts.append(str(val))
    text = "\n".join(parts)
    if strip_patterns:
        for pat in strip_patterns:
            text = pat.sub("", text)
        text = text.strip()
    return text


def _extract_label(record: dict, label_field: str) -> str:
    """Extract a label string from a record, or return empty string."""
    val = _resolve_dot_path(record, label_field)
    if val is _MISSING:
        return ""
    if isinstance(val, list):
        return str(val[0]) if val else ""
    return str(val)


# ── Probe implementation ────────────────────────────────────────────────

class JSONLProbe(BaseProbe):
    """Probe for JSONL / NDJSON files."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".jsonl") or lower.endswith(".ndjson")

    def run(self, file_path: str, **kwargs) -> ProbeResult:
        config = _parse_probe_config(kwargs.get("probe_config"))
        raw_text = safe_read_text(file_path)
        lines = raw_text.splitlines()

        # Parse all lines, collecting valid records + metadata
        records: list[dict] = []
        parse_errors = 0
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json_module.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    parse_errors += 1
            except (json_module.JSONDecodeError, ValueError):
                parse_errors += 1

        total_records = len(records)

        # Apply filters
        if config and config.filters:
            filter_set = RecordFilterSet.from_dicts(config.filters, config.filter_logic)
            records = [r for r in records if filter_set.match(r)]

        filtered_count = len(records)

        # Build chunks
        chunks = self._build_chunks(records, config, raw_text)

        # Stats
        approx_tokens = estimate_tokens(raw_text)
        stats: Dict[str, Any] = {
            "approx_tokens": approx_tokens,
            "total_lines": len(lines),
            "total_records": total_records,
            "filtered_records": filtered_count,
            "parse_errors": parse_errors,
        }

        # Infer schema from first record
        schema_sample = None
        if records:
            schema_sample = {k: type(v).__name__ for k, v in records[0].items()}

        # Topic
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=(
                f"JSONL file with {total_records} records"
                f" ({filtered_count} after filtering)."
            ),
        )

        # ToC: group by timestamp if available, otherwise by chunk index
        toc_entries = self._build_toc(chunks, records, config)

        structure = StructureInfo(
            toc=toc_entries,
            schema_tree=schema_sample,
        )

        # Small-file bypass
        if should_bypass(raw_text):
            stats["is_full_content"] = True
            if not chunks and raw_text.strip():
                chunks = [Chunk(content=raw_text, index=0, section="full")]
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type="jsonl",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=chunks,
                raw_snippet=raw_text,
            )
            brief, instr = self.generate_instructions(result)
            result.brief_intro = brief
            result.instructions = instr
            return result

        stats["is_full_content"] = False
        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="jsonl",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=raw_text[:500],
        )
        brief, instr = self.generate_instructions(result)
        result.brief_intro = brief
        result.instructions = instr
        return result

    def _build_chunks(
        self,
        records: list[dict],
        config: Optional[JSONLProbeConfig],
        raw_text: str,
    ) -> list[Chunk]:
        """Convert filtered records to chunks."""
        if not records:
            return []

        has_text_fields = config and config.text_fields
        label_field = config.label_field if config else None
        group_size = config.group_size if config else 1
        group_sep = config.group_separator if config else "\n"
        ts_field = config.timestamp_field if config else None

        # Compile strip patterns once
        strip_compiled: list[re.Pattern] | None = None
        if config and config.text_strip_patterns:
            strip_compiled = [re.compile(p, re.DOTALL) for p in config.text_strip_patterns]

        # Extract text for each record
        texts: list[str] = []
        for rec in records:
            if has_text_fields:
                text = _extract_text(rec, config.text_fields, strip_compiled)
            else:
                text = json_module.dumps(rec, ensure_ascii=False, default=str)

            if label_field:
                label = _extract_label(rec, label_field)
                if label:
                    text = f"{label}: {text}"

            texts.append(text)

        # Group into chunks
        chunks: list[Chunk] = []
        group_size = max(1, group_size)

        for i in range(0, len(texts), group_size):
            batch = texts[i : i + group_size]
            batch_records = records[i : i + group_size]

            content = group_sep.join(batch)
            if not content.strip():
                continue

            # Chunk metadata
            meta: Dict[str, Any] = {
                "record_start": i,
                "record_end": i + len(batch),
            }
            if ts_field:
                first_ts = _resolve_dot_path(batch_records[0], ts_field)
                if first_ts is not _MISSING:
                    meta["timestamp"] = first_ts

            section = f"records[{i}:{i + len(batch)}]"

            chunks.append(
                Chunk(
                    content=content,
                    index=len(chunks),
                    section=section,
                    metadata=meta,
                )
            )

        return chunks

    def _build_toc(
        self,
        chunks: list[Chunk],
        records: list[dict],
        config: Optional[JSONLProbeConfig],
    ) -> list[TocEntry]:
        """Build table of contents from chunks."""
        if not chunks:
            return []

        entries: list[TocEntry] = []

        # If we have many chunks, create summary entries (not one per chunk)
        if len(chunks) <= 20:
            for chunk in chunks:
                preview = chunk.content[:80] if chunk.content else ""
                ts = chunk.metadata.get("timestamp", "")
                annotation = f"ts: {ts}" if ts else None
                entries.append(
                    TocEntry(
                        level=1,
                        title=chunk.section or f"chunk[{chunk.index}]",
                        preview=preview,
                        annotation=annotation,
                    )
                )
        else:
            # Summarize: first, middle, last chunk + total count
            sample_indices = [0, len(chunks) // 2, len(chunks) - 1]
            for idx in sample_indices:
                c = chunks[idx]
                preview = c.content[:80] if c.content else ""
                entries.append(
                    TocEntry(
                        level=1,
                        title=c.section or f"chunk[{c.index}]",
                        preview=preview,
                        annotation=f"sample ({idx + 1}/{len(chunks)} total chunks)",
                    )
                )

        return entries

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        stats = probe_result.stats
        total = stats.get("total_records", 0)
        filtered = stats.get("filtered_records", 0)
        errors = stats.get("parse_errors", 0)

        brief = (
            f"JSONL file with {total} records"
            f" ({filtered} after filtering, {errors} parse errors)."
        )

        parts = [
            "Extraction Method:",
            "- Line-delimited JSON, one record per line",
            "- Records parsed individually (malformed lines skipped)",
        ]
        if filtered < total:
            parts.append(f"- {total - filtered} records filtered out by probe config")
        if stats.get("is_full_content"):
            parts.append("- Full content included (small file)")
        else:
            parts.append("- Content chunked by record groups")

        instructions = "\n".join(parts)
        return brief, instructions
