"""Indexing pipeline — file ingestion, probing, embedding, and storage."""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from mentat.core.telemetry import Telemetry
from mentat.core.queue import (
    ProcessingTask,
    build_chunk_records,
    prepare_embed_texts,
)
from mentat.probes import get_probe, run_probe
from mentat.probes.base import ProbeResult

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.indexer")


class Indexer:
    """Handles all document indexing operations."""

    def __init__(self, mentat: "Mentat"):
        self._m = mentat

    async def add(
        self,
        path: str,
        force: bool = False,
        summarize: bool = False,
        use_llm_instructions: bool = False,
        wait: bool = False,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        _logical_filename: Optional[str] = None,
        collection: Optional[str] = None,
        _skip_path_dedup: bool = False,
        probe_config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Index a file with async background processing.

        New async pipeline (default):
          1. Probe (Layer 2) — extract semantic fingerprint, no LLM (~1s)
          2. Generate template instructions (no LLM, instant)
          3. Store stub with ToC (no chunks yet)
          4. Queue background task for embedding + summarization
          5. Return immediately (or wait if wait=True)

        Background worker then:
          - Generates embeddings for all chunks
          - Optionally generates summaries (if summarize=True)
          - Stores chunks with vectors and summaries

        Args:
            path: File path to index
            force: Re-index even if content hash exists in cache
            summarize: Enable LLM-based chunk summarization (default: False, lazy)
            use_llm_instructions: Use LLM for instruction generation (default: False, template-based)
            wait: Wait for background processing to complete before returning (default: False)
            source: Origin tag for content provenance (e.g. "web_fetch", "composio:gmail")
            metadata: Arbitrary key-value metadata to store alongside the document

        Returns:
            Document ID. ToC is available immediately; full chunks available after background processing.

        Note:
            With wait=False (default), this returns in ~1-3s regardless of file size.
            Full vector search capability becomes available after background processing completes.
        """
        m = self._m

        # Check content hash cache
        if not force:
            cached_id = m.cache.get(path)
            if cached_id:
                logger.info(f"Cache hit for {path} -> {cached_id}, skipping.")
                print(
                    f"[Cache] Already indexed as {cached_id}. Use force=True to re-index."
                )
                # Update path index (may be missing for docs indexed before this feature)
                m.path_index.put(path, cached_id)
                # Still route to collections even on cache hit
                if source:
                    for coll_name in m.collections_store.get_auto_route_targets(source):
                        m.collections_store.add_doc(coll_name, cached_id)
                if collection:
                    m.collections_store.add_doc(collection, cached_id)
                return cached_id

        # Path-based dedup: if this path was previously indexed with different
        # content, clean up the old document before creating a new one.
        # Skipped when called from add_content() which handles its own dedup.
        if not _skip_path_dedup:
            old_doc_id = m.path_index.get(path)
            if old_doc_id:
                logger.info(f"Path {path} was previously indexed as {old_doc_id}, replacing.")
                m.storage.delete_doc(old_doc_id)
                m.cache.remove(old_doc_id)
                m.path_index.remove(old_doc_id)

        doc_id = str(uuid.uuid4())
        filename = _logical_filename or Path(path).name
        logger.info(f"Adding file: {path} (ID: {doc_id}, filename: {filename})")

        # Layer 2: Probe — extract skeleton (no LLM, fast ~1s)
        with Telemetry.time_it(doc_id, "probe"):
            probe_kwargs = {"probe_config": probe_config} if probe_config else {}
            probe_result = run_probe(path, **probe_kwargs)
            probe_result.doc_id = doc_id
            probe_result.filename = filename

        # Normalize chunk sizes for optimal retrieval performance
        from mentat.probes._utils import normalize_chunk_sizes
        probe_result.chunks = normalize_chunk_sizes(
            probe_result.chunks,
            target_tokens=m.config.chunk_target_tokens,
        )

        Telemetry.record_chunks(doc_id, len(probe_result.chunks))

        # Record fast mode (template + no summarization)
        fast_mode = not summarize and not use_llm_instructions
        if fast_mode:
            stats = Telemetry.get_stats(doc_id)
            if stats:
                stats.fast_mode = True

        # ── New async pipeline ───────────────────────────────────────
        # Generate instructions IMMEDIATELY (template-based, no LLM)
        with Telemetry.time_it(doc_id, "librarian"):
            brief_intro, instructions = m.librarian.generate_guide_template(
                probe_result
            )

        # Store stub with ToC (NO chunks yet — those are processed in background)
        m.storage.add_stub(
            doc_id=doc_id,
            filename=filename,
            brief_intro=brief_intro,
            instruction=instructions,
            probe_json=json.dumps(probe_result.model_dump(), default=str),
            source=source,
            metadata_json=json.dumps(metadata or {}, default=str),
        )
        logger.info(f"Stored stub for {filename}")

        # Raw file storage — save immediately for downstream access
        m.file_store.save(path, doc_id)

        # Record in content hash cache and path index
        m.cache.put(path, doc_id)
        m.path_index.put(path, doc_id)

        # Telemetry for immediate return
        original_size = os.path.getsize(path)
        stub_size = len(brief_intro.encode("utf-8"))
        savings = 1.0 - (stub_size / max(1, original_size))
        Telemetry.record_savings(doc_id, savings)

        # Adaptor hooks (called immediately, before chunks are processed)
        for adaptor in m._adaptors:
            adaptor.on_document_indexed(
                doc_id, {
                    "filename": filename,
                    "brief_intro": brief_intro,
                    "source": source,
                    "metadata": metadata or {},
                }
            )

        # ── Collection routing ────────────────────────────────────────
        # Auto-route: add to collections whose auto_add_sources match
        if source:
            for coll_name in m.collections_store.get_auto_route_targets(source):
                m.collections_store.add_doc(coll_name, doc_id)
        # Explicit collection
        if collection:
            m.collections_store.add_doc(collection, doc_id)

        # ── Chunk metadata fields for pre-filtering ─────────────────
        chunk_extra = {
            "source": source,
            "indexed_at": time.time(),
            "file_type": probe_result.file_type or "",
            "metadata_json": json.dumps(metadata or {}, default=str),
        }

        # ── Inline processing (wait=True) or queue (wait=False) ───────
        if wait:
            # Direct inline path — bypass queue for lower latency
            await self._process_chunks_inline(
                doc_id, filename, probe_result, summarize,
                extra_fields=chunk_extra,
            )
            logger.info(f"Successfully indexed {filename} (ID: {doc_id})")
            print(Telemetry.format_stats(doc_id))
        else:
            # Async queue path — return immediately
            task = ProcessingTask(
                doc_id=doc_id,
                probe_result=probe_result,
                priority=0,
                needs_summarization=summarize,
                chunk_extra_fields=chunk_extra,
            )
            await m.processor.queue.submit(task)
            logger.info(
                f"Queued background processing for {filename} (summarize={summarize})"
            )
            print(f"\u23f3 Queued: {filename} \u2192 {doc_id} (processing in background)")
            print(Telemetry.format_stats(doc_id))

        return doc_id

    async def _process_chunks_inline(
        self,
        doc_id: str,
        filename: str,
        probe_result: ProbeResult,
        summarize: bool,
        extra_fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Embed and store chunks directly (no queue, no polling).

        Used by ``add(wait=True)`` and ``add_batch()`` for lower latency.
        """
        m = self._m
        chunks = probe_result.chunks
        if not chunks:
            return

        embed_texts, chunk_mapping = prepare_embed_texts(chunks)

        async def _noop_summaries():
            return [""] * len(chunks)

        # Run embedding + optional summarization concurrently
        embed_coro = m.embeddings.embed_batch(embed_texts)
        if summarize:
            summ_coro = m.librarian.summarize_chunks(probe_result)
        else:
            summ_coro = _noop_summaries()

        vectors, summaries = await asyncio.gather(embed_coro, summ_coro)

        chunk_records = build_chunk_records(
            doc_id, filename, chunks, vectors, summaries, chunk_mapping,
            extra_fields=extra_fields,
        )

        if chunk_records:
            vector_dim = len(vectors[0])
            m.storage._ensure_chunks_table(vector_dim)
            m.storage.add_chunks(chunk_records)

        logger.debug(
            f"Inline: stored {len(chunk_records)} chunks for {doc_id}"
        )

    async def add_batch(
        self,
        paths: List[str],
        force: bool = False,
        summarize: bool = False,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Index multiple files efficiently with batched embedding.

        Probes all files first, collects all chunks, then embeds them in a
        single batched API call. Much faster than calling ``add(wait=True)``
        in a loop when indexing many files at once.

        Args:
            paths: List of file paths to index.
            force: Re-index even if cached.
            summarize: Enable LLM-based chunk summarization.
            source: Origin tag for content provenance (e.g. "web_fetch", "composio:gmail").
            metadata: Arbitrary key-value metadata to store alongside documents.

        Returns:
            List of document IDs in the same order as ``paths``.
        """
        m = self._m

        # Phase 1: Probe all files and store stubs
        doc_ids: List[str] = []
        probe_results: List[Optional[ProbeResult]] = []
        filenames: List[str] = []

        for path in paths:
            # Cache check
            if not force:
                cached_id = m.cache.get(path)
                if cached_id:
                    logger.info(f"Cache hit for {path} -> {cached_id}")
                    doc_ids.append(cached_id)
                    probe_results.append(None)
                    filenames.append(Path(path).name)
                    continue

            doc_id = str(uuid.uuid4())
            filename = Path(path).name

            with Telemetry.time_it(doc_id, "probe"):
                probe_result = run_probe(path)
                probe_result.doc_id = doc_id
                probe_result.filename = filename

            # Normalize chunk sizes for optimal retrieval performance
            from mentat.probes._utils import normalize_chunk_sizes
            probe_result.chunks = normalize_chunk_sizes(
                probe_result.chunks,
                target_tokens=m.config.chunk_target_tokens,
            )

            Telemetry.record_chunks(doc_id, len(probe_result.chunks))

            with Telemetry.time_it(doc_id, "librarian"):
                brief_intro, instructions = m.librarian.generate_guide_template(
                    probe_result
                )

            m.storage.add_stub(
                doc_id=doc_id,
                filename=filename,
                brief_intro=brief_intro,
                instruction=instructions,
                probe_json=json.dumps(probe_result.model_dump(), default=str),
                source=source,
                metadata_json=json.dumps(metadata or {}, default=str),
            )
            m.file_store.save(path, doc_id)
            m.cache.put(path, doc_id)

            original_size = os.path.getsize(path)
            stub_size = len(brief_intro.encode("utf-8"))
            Telemetry.record_savings(doc_id, 1.0 - (stub_size / max(1, original_size)))

            doc_ids.append(doc_id)
            probe_results.append(probe_result)
            filenames.append(filename)

        # Phase 2: Collect ALL chunks and embed in one batched call
        all_embed_texts: List[str] = []
        all_chunk_mappings: List[Any] = []
        file_boundaries: List[int] = []  # cumulative count of embed texts per file
        files_to_process: List[int] = []  # indices of files that need embedding

        for idx, pr in enumerate(probe_results):
            if pr is None or not pr.chunks:
                continue
            files_to_process.append(idx)
            texts, mapping = prepare_embed_texts(pr.chunks)
            all_embed_texts.extend(texts)
            all_chunk_mappings.append(mapping)
            file_boundaries.append(len(all_embed_texts))

        if all_embed_texts:
            logger.info(
                f"Batch embedding {len(all_embed_texts)} texts from "
                f"{len(files_to_process)} files"
            )

            # Single batched embedding call for all files
            all_vectors = await m.embeddings.embed_batch(all_embed_texts)

            # Optional: batch summarization concurrently
            if summarize:
                summ_tasks = [
                    m.librarian.summarize_chunks(probe_results[idx])
                    for idx in files_to_process
                ]
                all_summaries = await asyncio.gather(*summ_tasks)
            else:
                all_summaries = [
                    [""] * len(probe_results[idx].chunks)
                    for idx in files_to_process
                ]

            # Distribute vectors back to files and store
            vec_start = 0
            for i, file_idx in enumerate(files_to_process):
                vec_end = file_boundaries[i]
                pr = probe_results[file_idx]
                vectors = all_vectors[vec_start:vec_end]
                summaries = all_summaries[i]
                mapping = all_chunk_mappings[i]

                batch_extra = {
                    "source": source,
                    "indexed_at": time.time(),
                    "file_type": pr.file_type or "",
                    "metadata_json": json.dumps(metadata or {}, default=str),
                }
                records = build_chunk_records(
                    doc_ids[file_idx], filenames[file_idx],
                    pr.chunks, vectors, summaries, mapping,
                    extra_fields=batch_extra,
                )
                if records:
                    vector_dim = len(vectors[0])
                    m.storage._ensure_chunks_table(vector_dim)
                    m.storage.add_chunks(records)

                vec_start = vec_end

        return doc_ids

    async def add_content(
        self,
        content: str,
        filename: str,
        content_type: str = "text/plain",
        force: bool = False,
        summarize: bool = False,
        wait: bool = False,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        collection: Optional[str] = None,
    ) -> str:
        """Index raw content without requiring a file on disk.

        Writes content to a temp file in the storage directory, then
        delegates to the standard ``add()`` pipeline.  The content hash
        is computed from the string itself for deduplication.

        The caller-provided ``filename`` is preserved as the logical
        filename in storage (overriding the temp file name), so search
        results show meaningful names like "README.md" instead of hashes.

        Args:
            content: Raw text content to index.
            filename: Logical filename (used for probe format detection AND
                stored as the display name in search results).
            content_type: MIME type hint (currently unused, reserved).
            force: Re-index even if content hash matches an existing document.
            summarize: Enable LLM-based chunk summarization.
            wait: Block until background processing completes.
            source: Origin tag for content provenance (e.g. "web_fetch", "composio:gmail").
            metadata: Arbitrary key-value metadata to store alongside the document.

        Returns:
            Document ID.
        """
        m = self._m

        # Content-based dedup: hash the content string
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if not force:
            cached_id = m.cache.get_by_hash(content_hash)
            if cached_id:
                logger.info(f"Content cache hit for {filename} -> {cached_id}")
                # Still route to collections even on cache hit
                if source:
                    for coll_name in m.collections_store.get_auto_route_targets(source):
                        m.collections_store.add_doc(coll_name, cached_id)
                if collection:
                    m.collections_store.add_doc(collection, cached_id)
                return cached_id

        # Path-based dedup for content: use a synthetic key so that the same
        # logical filename replaces its previous version when content changes.
        content_path_key = f"__content__:{filename}"
        old_doc_id = m.path_index.get(content_path_key)
        if old_doc_id:
            logger.info(
                f"Content filename {filename} was previously indexed as {old_doc_id}, replacing."
            )
            m.storage.delete_doc(old_doc_id)
            m.cache.remove(old_doc_id)
            m.path_index.remove(old_doc_id)

        # Write content to a temp file so probes can read it.
        # Validate the suffix against registered probes; fall back to .md
        # (markdown probe handles plain text well) if no probe recognises it.
        suffix = Path(filename).suffix or ".md"
        if not get_probe(f"_probe_check{suffix}"):
            suffix = ".md"

        content_dir = Path(m.config.storage_dir) / "_content"
        content_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = content_dir / f"{content_hash[:16]}{suffix}"
        tmp_path.write_text(content, encoding="utf-8")

        try:
            doc_id = await self.add(
                str(tmp_path),
                force=force,
                summarize=summarize,
                wait=wait,
                source=source,
                metadata=metadata,
                _logical_filename=filename,
                collection=collection,
                _skip_path_dedup=True,
            )
            # Also store under the content hash for future dedup
            m.cache.put_hash(content_hash, doc_id)
            # Record content path key for path-based dedup on future updates
            m.path_index.put(content_path_key, doc_id)
            return doc_id
        finally:
            # Temp file can be cleaned up; raw file is stored by add()
            pass

    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """Get current processing status for a document."""
        m = self._m
        status = m.processor.queue.get_status(doc_id)

        if not status:
            stub = m.storage.get_stub(doc_id)
            if stub:
                return {
                    "doc_id": doc_id,
                    "status": "completed",
                    "submitted_at": None,
                    "error": None,
                    "needs_summarization": False,
                }
            else:
                return {
                    "doc_id": doc_id,
                    "status": "not_found",
                    "submitted_at": None,
                    "error": None,
                    "needs_summarization": False,
                }

        return status

    async def wait_for_completion(
        self, doc_id: str, timeout: float = 300
    ) -> bool:
        """Wait for a document's background processing to complete."""
        import time
        start = time.time()

        while time.time() - start < timeout:
            status_dict = self.get_processing_status(doc_id)
            status = status_dict.get("status")

            if status == "completed":
                return True
            elif status == "failed":
                logger.error(
                    f"Processing failed for {doc_id}: {status_dict.get('error')}"
                )
                return False
            elif status == "not_found":
                return True

            await asyncio.sleep(0.5)

        logger.warning(f"Timeout waiting for {doc_id} to complete")
        return False
