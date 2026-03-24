"""Document reading — inspect, read_segment, read_structured, summarization."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mentat.probes import run_probe
from mentat.probes.base import ProbeResult
from mentat.probes._utils import estimate_tokens

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.reader")


def _resolve_child_sections(toc_entries: List[dict], section_path: str) -> set:
    """Resolve a section name to itself plus all descendant section titles.

    Uses the ToC level hierarchy: if the matched entry is at level N,
    all consecutive entries at level > N are children until another
    entry at level <= N is found.

    Args:
        toc_entries: Flat list of ToC dicts with 'level' and 'title' keys.
        section_path: Section name to look up (case-insensitive).

    Returns:
        Set of section titles (original casing) that match the parent
        and all its children.  Empty set if no match found.
    """
    if not toc_entries:
        return set()

    section_lower = section_path.lower().strip()
    result: set = set()

    for i, entry in enumerate(toc_entries):
        title = (entry.get("title", "") or "").strip()
        title_lower = title.lower()

        if (
            title_lower == section_lower
            or section_lower in title_lower
            or title_lower in section_lower
        ):
            # Found a matching entry
            result.add(title)
            parent_level = entry.get("level", 1)

            # Collect all subsequent entries at deeper levels
            for j in range(i + 1, len(toc_entries)):
                child_entry = toc_entries[j]
                child_level = child_entry.get("level", 1)
                if child_level <= parent_level:
                    break  # Sibling or ancestor — stop
                child_title = (child_entry.get("title", "") or "").strip()
                if child_title:
                    result.add(child_title)

    return result


def _section_matches(chunk_section: str, section_set: set) -> bool:
    """Check if a chunk's section matches any entry in the filter set.

    Uses case-insensitive substring matching (same semantics as read_segment).
    """
    cs = (chunk_section or "").lower().strip()
    if not cs:
        return False
    return any(
        cs == s or s in cs or cs in s
        for s in section_set
    )


class Reader:
    """Handles document reading, inspection, and summarization."""

    def __init__(self, mentat: "Mentat"):
        self._m = mentat

    async def get_doc_meta(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Return lightweight document metadata.

        Consolidates brief_intro, instructions, toc_entries, source,
        metadata, and processing_status into a single response.

        Returns None if the document is not found.
        """
        m = self._m
        stub = m.storage.get_stub(doc_id)
        if not stub:
            return None

        # Extract ToC from probe_json
        toc_entries: List[dict] = []
        probe_json_str = stub.get("probe_json", "")
        if probe_json_str:
            try:
                probe_data = json.loads(probe_json_str)
                toc_entries = probe_data.get("structure", {}).get("toc", [])
            except (json.JSONDecodeError, TypeError):
                pass

        from mentat.core.searcher import Searcher
        status = m.indexer.get_processing_status(doc_id)

        return {
            "doc_id": doc_id,
            "filename": stub.get("filename"),
            "brief_intro": stub.get("brief_intro"),
            "instructions": stub.get("instruction"),
            "toc_entries": toc_entries,
            "source": stub.get("source", ""),
            "metadata": Searcher._parse_stub_metadata(stub),
            "processing_status": status.get("status", "unknown"),
        }

    async def inspect(
        self,
        doc_id: str,
        sections: Optional[List[str]] = None,
        full: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve document metadata, optionally with full probe data.

        Three modes:
          - **Lightweight** (default): doc_id, filename, brief_intro, source,
            metadata, toc.  Omits instruction, full probe, chunk_summaries.
          - **Section filter** (sections=["Methods"]): Lightweight plus
            instruction and chunk_summaries for the requested sections.
          - **Full** (full=True): Everything (current legacy behaviour).

        Args:
            doc_id: Document identifier.
            sections: Optional section names to filter chunk_summaries.
            full: If True, return the complete probe data and all chunks.
        """
        m = self._m
        from mentat.core.searcher import Searcher

        stub = m.storage.get_stub(doc_id)
        if not stub:
            return None

        # Always include lightweight base
        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "filename": stub.get("filename"),
            "brief_intro": stub.get("brief_intro"),
            "source": stub.get("source", ""),
            "metadata": Searcher._parse_stub_metadata(stub),
        }

        # Extract ToC from probe_json (always included)
        probe_json = stub.get("probe_json", "")
        toc_entries: List[dict] = []
        if probe_json:
            try:
                probe_data = json.loads(probe_json)
                toc_entries = probe_data.get("structure", {}).get("toc", [])
            except json.JSONDecodeError:
                pass
        result["toc"] = toc_entries

        # ── Full mode: include everything ──────────────────────────────
        if full:
            result["instruction"] = stub.get("instruction")
            if probe_json:
                try:
                    result["probe"] = json.loads(probe_json)
                except json.JSONDecodeError:
                    pass
            chunk_rows = m.storage.get_chunks_by_doc(doc_id)
            if chunk_rows:
                result["chunk_summaries"] = [
                    {
                        "index": r.get("chunk_index", 0),
                        "section": r.get("section", ""),
                        "summary": r.get("summary", ""),
                    }
                    for r in chunk_rows
                ]
            return result

        # ── Section filter mode ────────────────────────────────────────
        if sections:
            result["instruction"] = stub.get("instruction")
            # Resolve parent sections to include children (consistent with read_segment)
            expanded: set = set()
            for s in sections:
                expanded |= _resolve_child_sections(toc_entries, s)
            # Fallback to simple matching if ToC resolution found nothing
            if not expanded:
                expanded = {s.lower().strip() for s in sections}
                use_lowered = True
            else:
                use_lowered = False

            # Track section heat for inspected sections (weight 2.0)
            asyncio.create_task(
                m.section_heat.record_sections(doc_id, expanded, weight=2.0)
            )

            chunk_rows = m.storage.get_chunks_by_doc(doc_id)
            if chunk_rows:
                matched = []
                for r in chunk_rows:
                    cs = (r.get("section", "") or "").strip()
                    if not cs:
                        continue
                    if use_lowered:
                        hit = _section_matches(cs, expanded)
                    else:
                        cs_lower = cs.lower()
                        hit = any(
                            cs_lower == t.lower()
                            or t.lower() in cs_lower
                            or cs_lower in t.lower()
                            for t in expanded
                        )
                    if hit:
                        matched.append({
                            "index": r.get("chunk_index", 0),
                            "section": r.get("section", ""),
                            "summary": r.get("summary", ""),
                        })
                result["chunk_summaries"] = matched

        # ── Lightweight mode (default) — just toc + brief_intro ───────
        return result

    async def summarize_doc(self, doc_id: str) -> bool:
        """Generate summaries for a document's chunks (on-demand / lazy).

        Fetches stored chunks, generates LLM summaries, then persists them
        back via delete + re-add (LanceDB has no UPDATE).

        Returns True if summaries were generated and persisted, False otherwise.
        """
        m = self._m

        # Get stored chunks
        chunk_rows = m.storage.get_chunks_by_doc(doc_id)
        if not chunk_rows:
            return False

        # Check if summaries already exist (non-empty)
        if all(row.get("summary", "") for row in chunk_rows):
            logger.info(f"Document {doc_id} already has summaries")
            return False

        # Get probe result from stub
        stub = m.storage.get_stub(doc_id)
        if not stub:
            return False

        probe_json = stub.get("probe_json", "")
        if not probe_json:
            return False

        probe_result = ProbeResult.model_validate_json(probe_json)

        # Generate summaries
        logger.info(f"Generating summaries for {doc_id}...")
        chunk_summaries = await m.librarian.summarize_chunks(probe_result)

        # Persist: update each chunk row with its summary, then delete+re-add
        summary_map: Dict[int, str] = {}
        for i, s in enumerate(chunk_summaries):
            if i < len(probe_result.chunks):
                summary_map[probe_result.chunks[i].index] = s

        for row in chunk_rows:
            idx = row.get("chunk_index", 0)
            row["summary"] = summary_map.get(idx, row.get("summary", ""))

        m.storage.update_chunks(doc_id, chunk_rows)
        logger.info(
            f"Persisted {len(chunk_summaries)} summaries for {doc_id}"
        )

        return True

    async def read_structured(
        self,
        path: str,
        sections: Optional[List[str]] = None,
        include_content: bool = False,
    ) -> Dict[str, Any]:
        """Return a structured, token-efficient view of a file.

        For RAG consumers: instead of dumping raw file content, this returns
        the table of contents, brief summary, instructions, and optionally
        chunk summaries—typically 5-10x smaller than the raw file.

        Flow:
          1. If the file is already indexed, return inspect data with summaries.
          2. If not indexed, run a fast probe (no LLM) for ToC + brief_intro.
          3. Always track access (may trigger async embedding on repeat access).

        Args:
            path: Absolute or relative file path.
            sections: Optional list of section names to filter results.
            include_content: If True, include raw chunk text alongside summaries.

        Returns:
            Dict with doc_id, filename, file_type, brief_intro, toc, instructions,
            chunks, processing_status, and token_estimate.
        """
        m = self._m
        resolved = str(Path(path).resolve())

        # Track access (may promote to hot queue → auto-index)
        await m.track_access(resolved)

        # Check if already indexed (via content hash cache)
        cached_id = m.cache.get(resolved)

        if cached_id:
            # Already indexed — return rich data from storage
            status_info = m.indexer.get_processing_status(cached_id)
            processing_status = status_info.get("status", "completed")

            stub = m.storage.get_stub(cached_id)
            if not stub:
                processing_status = "not_indexed"
            else:
                result: Dict[str, Any] = {
                    "doc_id": cached_id,
                    "filename": stub.get("filename", Path(resolved).name),
                    "brief_intro": stub.get("brief_intro", ""),
                    "instructions": stub.get("instruction", ""),
                    "processing_status": processing_status,
                }

                # Parse probe JSON for ToC and file_type
                probe_json = stub.get("probe_json", "")
                if probe_json:
                    try:
                        probe_data = json.loads(probe_json)
                        result["file_type"] = probe_data.get("file_type", "unknown")
                        structure = probe_data.get("structure", {})
                        toc = structure.get("toc", [])
                        if sections:
                            section_set = set(s.lower() for s in sections)
                            toc = [
                                e for e in toc
                                if e.get("title", "").lower() in section_set
                            ]
                        result["toc"] = toc
                    except json.JSONDecodeError:
                        result["file_type"] = "unknown"
                        result["toc"] = []

                # Fetch chunk summaries if available
                chunk_rows = m.storage.get_chunks_by_doc(cached_id)
                if chunk_rows:
                    chunks_out = []
                    for r in chunk_rows:
                        chunk_section = r.get("section", "")
                        if sections:
                            section_set = set(s.lower() for s in sections)
                            if chunk_section.lower() not in section_set:
                                continue
                        entry: Dict[str, Any] = {
                            "section": chunk_section,
                            "summary": r.get("summary", ""),
                        }
                        if include_content:
                            entry["content"] = r.get("content", "")
                        chunks_out.append(entry)
                    result["chunks"] = chunks_out
                else:
                    result["chunks"] = []

                # Estimate token cost of this response
                result["token_estimate"] = estimate_tokens(
                    json.dumps(result, default=str)
                )
                return result

        # Not indexed — run fast probe (no LLM, ~1s)
        try:
            probe_result = run_probe(resolved)
        except ValueError:
            return {
                "doc_id": None,
                "filename": Path(resolved).name,
                "file_type": "unknown",
                "brief_intro": "",
                "toc": [],
                "instructions": "",
                "chunks": [],
                "processing_status": "unsupported",
                "token_estimate": 0,
            }

        brief_intro, instructions = m.librarian.generate_guide_template(probe_result)
        toc = [e.model_dump() for e in (probe_result.structure.toc or [])]

        if sections:
            section_set = set(s.lower() for s in sections)
            toc = [e for e in toc if e.get("title", "").lower() in section_set]

        chunks_out = []
        for chunk in probe_result.chunks:
            if sections:
                section_set = set(s.lower() for s in sections)
                if (chunk.section or "").lower() not in section_set:
                    continue
            entry = {"section": chunk.section or "", "summary": ""}
            if include_content:
                entry["content"] = chunk.content
            chunks_out.append(entry)

        result = {
            "doc_id": None,
            "filename": probe_result.filename or Path(resolved).name,
            "file_type": probe_result.file_type or "unknown",
            "brief_intro": brief_intro,
            "toc": toc,
            "instructions": instructions,
            "chunks": chunks_out,
            "processing_status": "not_indexed",
            "token_estimate": 0,
        }
        result["token_estimate"] = estimate_tokens(json.dumps(result, default=str))
        return result

    async def read_segment(
        self,
        doc_id: str,
        section_path: str,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """Read a specific section's content by doc_id and section name.

        This is step 2 of the two-step retrieval protocol:
        1. Agent calls search(toc_only=True) to discover documents + sections
        2. Agent calls read_segment(doc_id, section) to get specific content

        Section matching is case-insensitive substring match.  For parent
        sections (non-leaf ToC entries), all child sections are included
        automatically based on the ToC hierarchy.

        Args:
            doc_id: Document identifier from search results.
            section_path: Section name or partial match (e.g. "Installation"
                matches "Installation Guide").  Parent sections automatically
                include all child content.
            include_summary: Include chunk summaries alongside content.

        Returns:
            Dict with doc_id, filename, section_path, chunks, toc_context,
            token_estimate, and expanded (True if children were included).
        """
        m = self._m

        stub = m.storage.get_stub(doc_id)
        if not stub:
            return {
                "doc_id": doc_id,
                "error": "document_not_found",
                "chunks": [],
            }

        # Get stored chunks for this doc
        chunk_rows = m.storage.get_chunks_by_doc(doc_id)

        # Parse ToC from probe_json for parent-child resolution
        toc_entries: List[dict] = []
        probe_json_str = stub.get("probe_json", "")
        if probe_json_str:
            try:
                probe_data = json.loads(probe_json_str)
                toc_entries = probe_data.get("structure", {}).get("toc", [])
            except (json.JSONDecodeError, TypeError):
                pass

        # Resolve parent section to include children via ToC hierarchy
        expanded_sections = _resolve_child_sections(toc_entries, section_path)

        # Track section heat with parent→child propagation (weight 3.0)
        sections_to_track = expanded_sections if expanded_sections else {section_path}
        asyncio.create_task(
            m.section_heat.record_sections(doc_id, sections_to_track, weight=3.0)
        )

        # Match chunks by section name
        section_lower = section_path.lower().strip()
        matched_chunks = []

        for row in chunk_rows:
            chunk_section = (row.get("section", "") or "").strip()
            chunk_section_lower = chunk_section.lower()
            if not chunk_section_lower:
                continue  # Skip chunks with no section label

            matched = False
            if expanded_sections:
                # Match against expanded set (parent + children)
                matched = any(
                    chunk_section_lower == t.lower()
                    or t.lower() in chunk_section_lower
                    or chunk_section_lower in t.lower()
                    for t in expanded_sections
                )
            else:
                # Fallback: original substring matching (no ToC or no match)
                matched = (
                    chunk_section_lower == section_lower
                    or section_lower in chunk_section_lower
                    or chunk_section_lower in section_lower
                )

            if matched:
                entry: Dict[str, Any] = {
                    "chunk_index": row.get("chunk_index", 0),
                    "section": row.get("section", ""),
                    "content": row.get("content", ""),
                }
                if include_summary and row.get("summary"):
                    entry["summary"] = row["summary"]
                matched_chunks.append(entry)

        # Extract matching ToC entries for context
        toc_context: List[dict] = []
        if toc_entries and expanded_sections:
            for toc_entry in toc_entries:
                title = (toc_entry.get("title", "") or "").strip()
                if title and title in expanded_sections:
                    toc_context.append(toc_entry)
        elif toc_entries:
            for toc_entry in toc_entries:
                title = (toc_entry.get("title", "") or "").lower()
                if not title:
                    continue
                if (
                    title == section_lower
                    or section_lower in title
                    or title in section_lower
                ):
                    toc_context.append(toc_entry)

        is_expanded = len(expanded_sections) > 1 if expanded_sections else False

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "filename": stub.get("filename", ""),
            "section_path": section_path,
            "chunks": matched_chunks,
            "toc_context": toc_context,
            "token_estimate": estimate_tokens(
                json.dumps(matched_chunks, default=str)
            ),
            "expanded": is_expanded,
        }

        # If no chunks found, provide guidance
        if not matched_chunks:
            status = m.indexer.get_processing_status(doc_id)
            if status.get("status") in ("pending", "processing"):
                result["note"] = (
                    "Document is still being processed. "
                    "Chunks not yet available."
                )
            else:
                result["note"] = (
                    f"No chunks matched section '{section_path}'. "
                    "Use search(toc_only=True) to discover available sections."
                )

        return result
