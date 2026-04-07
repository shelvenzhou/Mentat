"""Search pipeline — query transformation, vector search, result assembly."""

import asyncio
import json
import logging
import math
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mentat.core.models import MentatResult, MentatDocResult, ChunkResult
from mentat.probes.base import TocEntry
from mentat.storage.filters import MetadataFilterSet

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.searcher")


def _extract_score(r: Dict[str, Any]) -> float:
    """Extract a distance-style score (lower = better) from a LanceDB result.

    Vector search returns ``_distance`` (lower = closer).
    Hybrid search returns ``_relevance_score`` (higher = more relevant).
    We normalise to "lower is better" so the rest of the pipeline can sort
    consistently.  When ``_relevance_score`` is present we invert it
    (``1 - score``) so a perfect FTS hit (1.0) becomes 0.0.
    """
    if "_mentat_final_score" in r:
        return r["_mentat_final_score"]
    if "_relevance_score" in r:
        return 1.0 - r["_relevance_score"]
    return r.get("_distance", 0.0)


def _minmax(values: List[float], *, invert: bool = False) -> List[float]:
    """Min-max normalise values to [0, 1]."""
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        base = [0.0 for _ in values]
    else:
        span = hi - lo
        base = [(v - lo) / span for v in values]
    if invert:
        return [1.0 - v for v in base]
    return base


class Searcher:
    """Handles all search and query operations."""

    def __init__(self, mentat: "Mentat"):
        self._m = mentat

    async def _raw_search(
        self,
        query: str,
        top_k: int,
        hybrid: bool,
        doc_ids: Optional[List[str]],
        source: Optional[str],
        collections: Optional[List[str]] = None,
        filters: Optional[MetadataFilterSet] = None,
    ) -> tuple:
        """Shared search pipeline: query transform, source/collection filter, embed, search.

        Returns:
            (raw_results, stub_cache, transformed_query)
            raw_results may be empty list; stub_cache maps doc_id -> stub dict.
        """
        m = self._m

        # Transform query via adaptors
        for adaptor in m._adaptors:
            query = adaptor.transform_query(query)

        # Collection filtering: union of doc_ids from all named collections
        if collections:
            coll_ids: set = set()
            for coll_name in collections:
                coll_ids.update(m.collections_store.get_doc_ids(coll_name))
            if not coll_ids:
                return [], {}, query
            if doc_ids is not None:
                doc_ids = [d for d in doc_ids if d in coll_ids]
            else:
                doc_ids = list(coll_ids)
            if not doc_ids:
                return [], {}, query

        # Source filtering: resolve matching doc_ids and merge with explicit doc_ids
        if source:
            source_doc_ids = m.storage.get_doc_ids_by_source(source)
            if not source_doc_ids:
                return [], {}, query
            if doc_ids is not None:
                allowed = set(source_doc_ids)
                doc_ids = [d for d in doc_ids if d in allowed]
                if not doc_ids:
                    return [], {}, query
            else:
                doc_ids = source_doc_ids

        candidate_limit = top_k
        if m.reranker is not None and m.config.reranker_enabled:
            candidate_limit = max(
                top_k,
                m.config.reranker_top_n,
                top_k + 5,
                top_k * max(1, m.config.reranker_candidate_multiplier),
            )
        elif m.config.search_heat_weight > 0:
            # Heat bias needs a small over-fetch window so it can promote
            # recently-used sections that would otherwise miss the final cut.
            candidate_limit = max(top_k, top_k + 5)

        query_vector = await m.embeddings.embed(query)
        raw_results = m.storage.search(
            query_vector, query, limit=candidate_limit, use_hybrid=hybrid, doc_ids=doc_ids,
            filters=filters,
        )

        if not raw_results:
            return [], {}, query

        # Batch-fetch stubs for all unique doc_ids
        unique_doc_ids = {r.get("doc_id", "") for r in raw_results}
        stub_cache: Dict[str, Dict] = {}
        for did in unique_doc_ids:
            if did:
                stub_cache[did] = m.storage.get_stub(did) or {}

        raw_results = await self._rerank_results(query, raw_results)
        raw_results = self._apply_heat_bias(raw_results)
        return raw_results[:top_k], stub_cache, query

    async def _rerank_results(
        self,
        query: str,
        raw_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply cross-encoder reranking to the candidate set."""
        m = self._m
        if not raw_results:
            return []
        if m.reranker is None or not m.config.reranker_enabled:
            ranked = sorted(raw_results, key=_extract_score)
            for row in ranked:
                row["_mentat_final_score"] = _extract_score(row)
            return ranked

        logger.debug(
            "Applying reranker: provider=%s candidates=%d",
            m.config.reranker_provider,
            len(raw_results),
        )

        pairs = []
        for row in raw_results:
            text = row.get("content") or row.get("summary") or ""
            pairs.append((query, text))

        rerank_scores = await m.reranker.score_pairs(pairs)
        base_scores = [_extract_score(r) for r in raw_results]
        norm_base = _minmax(base_scores, invert=False)
        norm_rerank = _minmax(list(rerank_scores), invert=True)
        weight = min(max(m.config.reranker_weight, 0.0), 1.0)

        ranked_rows: List[Dict[str, Any]] = []
        for row, base_score, rerank_score, base_norm, rerank_norm in zip(
            raw_results, base_scores, rerank_scores, norm_base, norm_rerank
        ):
            merged = ((1.0 - weight) * base_norm) + (weight * rerank_norm)
            updated = dict(row)
            updated["_mentat_base_score"] = base_score
            updated["_mentat_rerank_score"] = float(rerank_score)
            updated["_mentat_final_score"] = merged
            ranked_rows.append(updated)

        ranked_rows.sort(key=_extract_score)
        return ranked_rows

    def _apply_heat_bias(
        self,
        raw_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Apply decayed section heat as an independent search bias."""
        weight = self._m.config.search_heat_weight
        if not raw_results or weight <= 0:
            return raw_results

        adjusted: List[Dict[str, Any]] = []
        for row in raw_results:
            updated = dict(row)
            base_score = updated.get("_mentat_final_score", _extract_score(updated))
            heat_score = 0.0
            section = updated.get("section")
            doc_id = updated.get("doc_id", "")
            if section and doc_id:
                heat_score = self._m.section_heat.get_score(doc_id, section)
            heat_bias = weight * (heat_score / (1.0 + heat_score))
            updated["_mentat_heat_score"] = heat_score
            updated["_mentat_final_score"] = base_score - heat_bias
            adjusted.append(updated)

        adjusted.sort(key=_extract_score)
        return adjusted

    @staticmethod
    def _parse_toc_entries(stub: Dict, with_metadata: bool) -> List[TocEntry]:
        """Extract TocEntry list from a stub's probe_json."""
        if not with_metadata:
            return []
        probe_json_str = stub.get("probe_json", "")
        if not probe_json_str:
            return []
        try:
            probe_data = json.loads(probe_json_str)
            raw = probe_data.get("structure", {}).get("toc", [])
            return [TocEntry(**e) if isinstance(e, dict) else e for e in raw]
        except (json.JSONDecodeError, TypeError):
            return []

    def _get_status_note(self, doc_id: str, has_active_tasks: bool) -> str:
        """Return processing status note for a document (empty if not active)."""
        if not has_active_tasks:
            return ""
        m = self._m
        status_info = m.processor.queue.get_status(doc_id)
        if status_info and status_info.get("status") in ("pending", "processing"):
            m.processor.queue.bump_priority(doc_id, delta=10)
            status_val = status_info["status"]
            if status_val == "pending":
                return "\n\n\u23f3 [Processing: This document is queued for embedding and will be fully searchable soon]"
            elif status_val == "processing":
                return "\n\n\U0001f504 [Processing: This document is currently being indexed and will be fully available shortly]"
        return ""

    @staticmethod
    def _parse_stub_metadata(stub: Dict) -> Dict[str, Any]:
        """Parse metadata_json from a stub dict."""
        stub_metadata: Dict[str, Any] = {}
        stub_metadata_json = stub.get("metadata_json", "")
        if stub_metadata_json and stub_metadata_json != "{}":
            try:
                stub_metadata = json.loads(stub_metadata_json)
            except (json.JSONDecodeError, TypeError):
                pass
        return stub_metadata

    async def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
        toc_only: bool = False,
        source: Optional[str] = None,
        with_metadata: Optional[bool] = None,
        collections: Optional[List[str]] = None,
        filters: Optional[MetadataFilterSet] = None,
    ) -> List[MentatResult]:
        """Search for relevant chunks and return results with instructions.

        Each result includes the chunk content, its section context,
        and optionally the document-level brief_intro + instructions.

        Args:
            doc_ids: If provided, restrict search to these documents only.
            toc_only: If True, return one result per document with ToC entries
                and matched section names instead of full chunk content.
                This is step 1 of the two-step retrieval protocol.
            source: Filter results by source tag. Supports glob suffix
                (e.g. "composio:*" matches all Composio sources).
            with_metadata: Include brief_intro, instructions, and toc_entries
                in results.  Defaults to True when toc_only=True (discovery
                needs metadata), False when toc_only=False (agent already
                knows the doc).
            collections: If provided, restrict search to docs in these
                collections (OR semantics — union of all named collections).
        """
        m = self._m

        # Default: with_metadata follows toc_only if not explicitly set
        if with_metadata is None:
            with_metadata = toc_only

        raw_results, stub_cache, query = await self._raw_search(
            query, top_k, hybrid, doc_ids, source, collections, filters=filters
        )
        if not raw_results:
            return []

        # ToC-only mode: group by document, return summaries instead of chunks
        if toc_only:
            results = self._build_toc_results(raw_results, stub_cache, top_k, with_metadata)
        else:
            # Only check queue status if the processor has pending/processing tasks
            has_active_tasks = bool(m.processor.queue._tasks)

            results = []
            for r in raw_results:
                doc_id = r.get("doc_id", "")
                stub = stub_cache.get(doc_id, {})
                base_instructions = stub.get("instruction", "") if with_metadata else ""
                status_note = self._get_status_note(doc_id, has_active_tasks)
                instructions = base_instructions + status_note

                results.append(
                    MentatResult(
                        doc_id=doc_id,
                        chunk_id=r.get("chunk_id", ""),
                        filename=r.get("filename", ""),
                        section=r.get("section"),
                        content=r.get("content", ""),
                        summary=r.get("summary", ""),
                        brief_intro=stub.get("brief_intro", "") if with_metadata else "",
                        instructions=instructions,
                        score=_extract_score(r),
                        source=stub.get("source", ""),
                        metadata=self._parse_stub_metadata(stub),
                    )
                )

            # Adaptor hooks (skip serialization if no adaptors)
            if m._adaptors:
                result_dicts = [r.model_dump() for r in results]
                for adaptor in m._adaptors:
                    result_dicts = adaptor.on_search_results(query, result_dicts)

        # Track section heat from search results (weight 1.0)
        _heat_by_doc: Dict[str, set] = {}
        for r in results:
            if r.section and r.doc_id:
                # toc_only mode joins sections with ", "
                for s in r.section.split(", "):
                    s = s.strip()
                    if s:
                        _heat_by_doc.setdefault(r.doc_id, set()).add(s)
        for did, secs in _heat_by_doc.items():
            asyncio.create_task(
                m.section_heat.record_sections(did, secs, weight=1.0)
            )

        return results

    async def search_grouped(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
        toc_only: bool = False,
        source: Optional[str] = None,
        with_metadata: Optional[bool] = None,
        collections: Optional[List[str]] = None,
        filters: Optional[MetadataFilterSet] = None,
    ) -> List[MentatDocResult]:
        """Search and return results grouped by document.

        Same semantics as search() but groups chunks under their parent
        document to eliminate duplicate metadata.  Each MentatDocResult
        contains doc-level fields once and a list of ChunkResult items.

        Args:
            Same as search().
        """
        m = self._m

        if with_metadata is None:
            with_metadata = toc_only

        raw_results, stub_cache, query = await self._raw_search(
            query, top_k, hybrid, doc_ids, source, collections, filters=filters
        )
        if not raw_results:
            return []

        has_active_tasks = bool(m.processor.queue._tasks)

        # Group raw results by doc_id
        doc_groups: Dict[str, Dict[str, Any]] = {}
        for r in raw_results:
            did = r.get("doc_id", "")
            score = _extract_score(r)
            if did not in doc_groups:
                doc_groups[did] = {"chunks": [], "best_score": score, "sections": set()}
            doc_groups[did]["chunks"].append(r)
            doc_groups[did]["best_score"] = min(doc_groups[did]["best_score"], score)
            section = r.get("section", "")
            if section:
                doc_groups[did]["sections"].add(section)

        results = []
        for did, group in doc_groups.items():
            stub = stub_cache.get(did, {})
            base_instructions = stub.get("instruction", "") if with_metadata else ""
            status_note = self._get_status_note(did, has_active_tasks)
            instructions = base_instructions + status_note

            toc_entries = self._parse_toc_entries(stub, with_metadata)

            # Build chunk list (empty in toc_only mode)
            chunk_results: List[ChunkResult] = []
            if not toc_only:
                for r in group["chunks"]:
                    chunk_results.append(
                        ChunkResult(
                            chunk_id=r.get("chunk_id", ""),
                            section=r.get("section"),
                            content=r.get("content", ""),
                            summary=r.get("summary", ""),
                            score=_extract_score(r),
                        )
                    )

            results.append(
                MentatDocResult(
                    doc_id=did,
                    filename=stub.get("filename", ""),
                    brief_intro=stub.get("brief_intro", "") if with_metadata else "",
                    instructions=instructions,
                    source=stub.get("source", ""),
                    metadata=self._parse_stub_metadata(stub),
                    toc_entries=toc_entries,
                    score=group["best_score"],
                    chunks=chunk_results,
                )
            )

        # Track section heat from grouped search results (weight 1.0)
        for did, group in doc_groups.items():
            secs = group.get("sections", set())
            if secs:
                asyncio.create_task(
                    m.section_heat.record_sections(did, secs, weight=1.0)
                )

        results.sort(key=lambda r: r.score)
        return results[:top_k]

    def _build_toc_results(
        self,
        raw_results: List[Dict[str, Any]],
        stub_cache: Dict[str, Dict],
        top_k: int,
        with_metadata: bool = True,
    ) -> List[MentatResult]:
        """Group chunk-level search results by document, returning ToC summaries.

        Used in toc_only mode: one MentatResult per document with matched
        section names and full ToC entries instead of chunk content.

        Args:
            with_metadata: Include brief_intro, instructions, and toc_entries.
        """
        m = self._m
        doc_groups: Dict[str, Dict[str, Any]] = {}

        for r in raw_results:
            doc_id = r.get("doc_id", "")
            section = r.get("section", "")
            score = _extract_score(r)

            if doc_id not in doc_groups:
                doc_groups[doc_id] = {"sections": set(), "best_score": score}
            if section:
                doc_groups[doc_id]["sections"].add(section)
            doc_groups[doc_id]["best_score"] = min(
                doc_groups[doc_id]["best_score"], score
            )

        # Only check queue status if the processor has pending/processing tasks
        has_active_tasks = bool(m.processor.queue._tasks)

        results = []
        for doc_id, group in doc_groups.items():
            stub = stub_cache.get(doc_id, {})

            toc_entries = self._parse_toc_entries(stub, with_metadata)

            matched_sections = sorted(group["sections"])
            base_instructions = stub.get("instruction", "") if with_metadata else ""
            status_note = self._get_status_note(doc_id, has_active_tasks)
            instructions = base_instructions + status_note

            results.append(
                MentatResult(
                    doc_id=doc_id,
                    chunk_id="",
                    filename=stub.get("filename", ""),
                    section=", ".join(matched_sections) if matched_sections else None,
                    content="",
                    summary="",
                    brief_intro=stub.get("brief_intro", "") if with_metadata else "",
                    instructions=instructions,
                    score=group["best_score"],
                    toc_entries=toc_entries,
                    source=stub.get("source", ""),
                    metadata=self._parse_stub_metadata(stub),
                )
            )

        results.sort(key=lambda r: r.score)
        return results[:top_k]
