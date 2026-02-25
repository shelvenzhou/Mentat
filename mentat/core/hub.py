import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel

from dotenv import load_dotenv

from mentat.core.telemetry import Telemetry
from mentat.core.embeddings import EmbeddingRegistry
from mentat.core.queue import (
    BackgroundProcessor,
    ProcessingTask,
    build_chunk_records,
    prepare_embed_texts,
)
from mentat.probes import get_probe, run_probe
from mentat.probes.base import ProbeResult
from mentat.probes._utils import estimate_tokens
from mentat.librarian.engine import Librarian
from mentat.storage.vector_db import LanceDBStorage
from mentat.storage.file_store import LocalFileStore
from mentat.storage.cache import ContentHashCache
from mentat.storage.collections import CollectionStore
from mentat.adaptors import BaseAdaptor

# Load .env at import time so all env vars (including API keys for litellm)
# are available before any LLM/embedding calls.
load_dotenv()


class MentatResult(BaseModel):
    """A search result returned by Mentat."""

    doc_id: str
    chunk_id: str
    filename: str
    section: Optional[str] = None
    content: str = ""
    summary: str = ""
    brief_intro: str = ""
    instructions: str = ""
    score: float = 0.0


@dataclass
class MentatConfig:
    """Configuration for Mentat instance.

    Values are resolved in order: explicit argument > environment variable > default.
    Environment variables use the ``MENTAT_`` prefix (see .env.example).
    """

    # Storage
    db_path: str = ""
    storage_dir: str = ""

    # Summary model — Phase 1: bulk chunk summarisation (use a fast/cheap model)
    summary_model: str = ""
    summary_api_key: str = ""
    summary_api_base: str = ""

    # Embedding model
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_api_base: str = ""

    # Background processing
    max_concurrent_tasks: int = 5

    # Chunk normalization
    chunk_target_tokens: int = 1000

    def __post_init__(self):
        self.db_path = self.db_path or os.getenv("MENTAT_DB_PATH", "./mentat_db")
        self.storage_dir = self.storage_dir or os.getenv(
            "MENTAT_STORAGE_DIR", "./mentat_files"
        )

        # Summary model (fast/cheap — bulk chunk summarisation)
        self.summary_model = self.summary_model or os.getenv(
            "MENTAT_SUMMARY_MODEL", "gpt-4o-mini"
        )
        self.summary_api_key = self.summary_api_key or os.getenv(
            "MENTAT_SUMMARY_API_KEY", ""
        )
        self.summary_api_base = self.summary_api_base or os.getenv(
            "MENTAT_SUMMARY_API_BASE", ""
        )

        # Embedding
        self.embedding_provider = self.embedding_provider or os.getenv(
            "MENTAT_EMBEDDING_PROVIDER", "litellm"
        )
        self.embedding_model = self.embedding_model or os.getenv(
            "MENTAT_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embedding_api_key = self.embedding_api_key or os.getenv(
            "MENTAT_EMBEDDING_API_KEY", ""
        )
        self.embedding_api_base = self.embedding_api_base or os.getenv(
            "MENTAT_EMBEDDING_API_BASE", ""
        )

        # Background processing
        self.max_concurrent_tasks = int(
            os.getenv("MENTAT_MAX_CONCURRENT_TASKS", str(self.max_concurrent_tasks))
        )

        # Chunk normalization
        self.chunk_target_tokens = int(
            os.getenv("MENTAT_CHUNK_TARGET_TOKENS", str(self.chunk_target_tokens))
        )


class Mentat:
    """Core orchestrator for the Mentat system.

    Usage:
        import mentat
        doc_id = await mentat.add("paper.pdf")
        results = await mentat.search("What algorithm?")
        probe_result = mentat.probe("data.csv")
    """

    _instance: Optional["Mentat"] = None

    def __init__(self, config: Optional[MentatConfig] = None):
        self.config = config or MentatConfig()
        self.logger = logging.getLogger("mentat")

        # Layer 1: Storage (Haystack)
        self.storage = LanceDBStorage(db_path=self.config.db_path)
        self.file_store = LocalFileStore(storage_dir=self.config.storage_dir)
        self.cache = ContentHashCache(cache_dir=self.config.db_path)

        # Layer 3: Librarian
        self.librarian = Librarian(
            summary_model=self.config.summary_model,
            summary_api_key=self.config.summary_api_key or None,
            summary_api_base=self.config.summary_api_base or None,
        )

        # Embeddings
        self.embeddings = EmbeddingRegistry.get_provider(
            self.config.embedding_provider,
            model=self.config.embedding_model,
            api_key=self.config.embedding_api_key or None,
            api_base=self.config.embedding_api_base or None,
        )

        # Collections
        self.collections_store = CollectionStore(store_dir=self.config.db_path)

        # Adaptors
        self._adaptors: List[BaseAdaptor] = []

        # Background processor (async queue system)
        self.processor = BackgroundProcessor(self, max_concurrent=self.config.max_concurrent_tasks)

    @classmethod
    def get_instance(cls, config: Optional[MentatConfig] = None) -> "Mentat":
        """Singleton accessor for module-level API."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (useful for tests)."""
        cls._instance = None

    def register_adaptor(self, adaptor: BaseAdaptor):
        """Register an adaptor for lifecycle hooks."""
        self._adaptors.append(adaptor)

    async def start(self):
        """Start the background processor.

        Should be called once on application startup to enable async processing.
        If not called, documents will queue but not process until start() is invoked.
        """
        await self.processor.start()

    async def shutdown(self):
        """Shutdown the background processor gracefully.

        Waits for currently processing tasks to complete before stopping.
        Should be called on application shutdown.
        """
        await self.processor.stop()

    async def add(
        self,
        path: str,
        force: bool = False,
        summarize: bool = False,
        use_llm_instructions: bool = False,
        wait: bool = False,
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

        Returns:
            Document ID. ToC is available immediately; full chunks available after background processing.

        Note:
            With wait=False (default), this returns in ~1-3s regardless of file size.
            Full vector search capability becomes available after background processing completes.
        """
        # Check content hash cache
        if not force:
            cached_id = self.cache.get(path)
            if cached_id:
                self.logger.info(f"Cache hit for {path} -> {cached_id}, skipping.")
                print(
                    f"[Cache] Already indexed as {cached_id}. Use force=True to re-index."
                )
                return cached_id

        doc_id = str(uuid.uuid4())
        filename = Path(path).name
        self.logger.info(f"Adding file: {path} (ID: {doc_id})")

        # Layer 2: Probe — extract skeleton (no LLM, fast ~1s)
        with Telemetry.time_it(doc_id, "probe"):
            probe_result = run_probe(path)
            probe_result.doc_id = doc_id
            probe_result.filename = filename

        # Normalize chunk sizes for optimal retrieval performance
        from mentat.probes._utils import normalize_chunk_sizes
        probe_result.chunks = normalize_chunk_sizes(
            probe_result.chunks,
            target_tokens=self.config.chunk_target_tokens,
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
        # Note: use_llm_instructions is ignored for now (always use template for immediate return)
        # LLM-based instruction generation could be added to background queue in future
        with Telemetry.time_it(doc_id, "librarian"):
            brief_intro, instructions = self.librarian.generate_guide_template(
                probe_result
            )

        # Store stub with ToC (NO chunks yet — those are processed in background)
        self.storage.add_stub(
            doc_id=doc_id,
            filename=filename,
            brief_intro=brief_intro,
            instruction=instructions,
            probe_json=json.dumps(probe_result.model_dump(), default=str),
        )
        self.logger.info(f"Stored stub for {filename}")

        # Raw file storage — save immediately for downstream access
        self.file_store.save(path, doc_id)

        # Record in content hash cache
        self.cache.put(path, doc_id)

        # Telemetry for immediate return
        original_size = os.path.getsize(path)
        stub_size = len(brief_intro.encode("utf-8"))
        savings = 1.0 - (stub_size / max(1, original_size))
        Telemetry.record_savings(doc_id, savings)

        # Adaptor hooks (called immediately, before chunks are processed)
        for adaptor in self._adaptors:
            adaptor.on_document_indexed(
                doc_id, {"filename": filename, "brief_intro": brief_intro}
            )

        # ── Inline processing (wait=True) or queue (wait=False) ───────
        if wait:
            # Direct inline path — bypass queue for lower latency
            await self._process_chunks_inline(
                doc_id, filename, probe_result, summarize
            )
            self.logger.info(f"Successfully indexed {filename} (ID: {doc_id})")
            print(Telemetry.format_stats(doc_id))
        else:
            # Async queue path — return immediately
            task = ProcessingTask(
                doc_id=doc_id,
                probe_result=probe_result,
                priority=0,
                needs_summarization=summarize,
            )
            await self.processor.queue.submit(task)
            self.logger.info(
                f"Queued background processing for {filename} (summarize={summarize})"
            )
            print(f"⏳ Queued: {filename} → {doc_id} (processing in background)")
            print(Telemetry.format_stats(doc_id))

        return doc_id

    async def _process_chunks_inline(
        self,
        doc_id: str,
        filename: str,
        probe_result: ProbeResult,
        summarize: bool,
    ) -> None:
        """Embed and store chunks directly (no queue, no polling).

        Used by ``add(wait=True)`` and ``add_batch()`` for lower latency.
        """
        chunks = probe_result.chunks
        if not chunks:
            return

        embed_texts, chunk_mapping = prepare_embed_texts(chunks)

        async def _noop_summaries():
            return [""] * len(chunks)

        # Run embedding + optional summarization concurrently
        embed_coro = self.embeddings.embed_batch(embed_texts)
        if summarize:
            summ_coro = self.librarian.summarize_chunks(probe_result)
        else:
            summ_coro = _noop_summaries()

        vectors, summaries = await asyncio.gather(embed_coro, summ_coro)

        chunk_records = build_chunk_records(
            doc_id, filename, chunks, vectors, summaries, chunk_mapping
        )

        if chunk_records:
            vector_dim = len(vectors[0])
            self.storage._ensure_chunks_table(vector_dim)
            self.storage.add_chunks(chunk_records)

        self.logger.debug(
            f"Inline: stored {len(chunk_records)} chunks for {doc_id}"
        )

    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        """Get current processing status for a document.

        Args:
            doc_id: Document identifier

        Returns:
            Status dict with keys:
                - doc_id: Document ID
                - status: "pending" | "processing" | "completed" | "failed" | "not_found"
                - submitted_at: Timestamp when queued (if in queue)
                - error: Error message (if failed)
                - needs_summarization: Whether summarization was requested

        Note:
            Status is tracked in-memory only. Documents indexed before the queue
            system or not found in the queue return status="not_found".
        """
        status = self.processor.queue.get_status(doc_id)

        if not status:
            # Check if document exists in storage (legacy or not in queue)
            stub = self.storage.get_stub(doc_id)
            if stub:
                # Document exists but not in queue - assume completed
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
        """Wait for a document's background processing to complete.

        Args:
            doc_id: Document identifier
            timeout: Maximum wait time in seconds (default: 300 = 5 minutes)

        Returns:
            True if processing completed successfully, False if timeout or failed

        Note:
            Polls the processing status every 0.5 seconds. If the document
            is not in the queue, returns True immediately (assumes already processed).
        """
        start = time.time()

        while time.time() - start < timeout:
            status_dict = self.get_processing_status(doc_id)
            status = status_dict.get("status")

            if status == "completed":
                return True
            elif status == "failed":
                self.logger.error(
                    f"Processing failed for {doc_id}: {status_dict.get('error')}"
                )
                return False
            elif status == "not_found":
                # Not in queue - either already processed or never queued
                return True

            # Still pending or processing - wait and retry
            await asyncio.sleep(0.5)

        # Timeout
        self.logger.warning(f"Timeout waiting for {doc_id} to complete")
        return False

    async def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[MentatResult]:
        """Search for relevant chunks and return results with instructions.

        Each result includes the chunk content, its section context,
        and the document-level brief_intro + instructions from the Librarian.

        Args:
            doc_ids: If provided, restrict search to these documents only.
        """
        # Transform query via adaptors
        for adaptor in self._adaptors:
            query = adaptor.transform_query(query)

        query_vector = await self.embeddings.embed(query)
        raw_results = self.storage.search(
            query_vector, query, limit=top_k, use_hybrid=hybrid, doc_ids=doc_ids
        )

        if not raw_results:
            return []

        # Batch-fetch stubs for all unique doc_ids
        unique_doc_ids = {r.get("doc_id", "") for r in raw_results}
        stub_cache: Dict[str, Dict] = {}
        for did in unique_doc_ids:
            if did:
                stub_cache[did] = self.storage.get_stub(did) or {}

        # Only check queue status if the processor has pending/processing tasks
        has_active_tasks = bool(self.processor.queue._tasks)

        results = []
        for r in raw_results:
            doc_id = r.get("doc_id", "")
            stub = stub_cache.get(doc_id, {})
            instructions = stub.get("instruction", "")

            # Only check queue status if there are active tasks (fast path)
            if has_active_tasks:
                status_info = self.processor.queue.get_status(doc_id)
                if status_info and status_info.get("status") in ("pending", "processing"):
                    self.processor.queue.bump_priority(doc_id, delta=10)
                    status_val = status_info["status"]
                    if status_val == "pending":
                        instructions += "\n\n⏳ [Processing: This document is queued for embedding and will be fully searchable soon]"
                    elif status_val == "processing":
                        instructions += "\n\n🔄 [Processing: This document is currently being indexed and will be fully available shortly]"

            results.append(
                MentatResult(
                    doc_id=doc_id,
                    chunk_id=r.get("chunk_id", ""),
                    filename=r.get("filename", ""),
                    section=r.get("section"),
                    content=r.get("content", ""),
                    summary=r.get("summary", ""),
                    brief_intro=stub.get("brief_intro", ""),
                    instructions=instructions,
                    score=r.get("_distance", 0.0),
                )
            )

        # Adaptor hooks (skip serialization if no adaptors)
        if self._adaptors:
            result_dicts = [r.model_dump() for r in results]
            for adaptor in self._adaptors:
                result_dicts = adaptor.on_search_results(query, result_dicts)

        return results

    async def inspect(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full probe results, instructions, and chunk summaries."""
        stub = self.storage.get_stub(doc_id)
        if not stub:
            return None

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "filename": stub.get("filename"),
            "brief_intro": stub.get("brief_intro"),
            "instruction": stub.get("instruction"),
        }

        # Parse probe JSON if available
        probe_json = stub.get("probe_json", "")
        if probe_json:
            try:
                result["probe"] = json.loads(probe_json)
            except json.JSONDecodeError:
                pass

        # Fetch stored chunk summaries
        chunk_rows = self.storage.get_chunks_by_doc(doc_id)
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

    def collection(self, name: str) -> "Collection":
        """Get a Collection handle for scoped add/search operations."""
        return Collection(name, self)

    def list_collections(self) -> List[str]:
        """Return all collection names."""
        return self.collections_store.list_collections()

    async def summarize_doc(self, doc_id: str) -> bool:
        """Generate summaries for a document's chunks (on-demand / lazy).

        Fetches stored chunks, generates LLM summaries, then persists them
        back via delete + re-add (LanceDB has no UPDATE).

        Returns True if summaries were generated and persisted, False otherwise.
        """
        # Get stored chunks
        chunk_rows = self.storage.get_chunks_by_doc(doc_id)
        if not chunk_rows:
            return False

        # Check if summaries already exist (non-empty)
        if all(row.get("summary", "") for row in chunk_rows):
            self.logger.info(f"Document {doc_id} already has summaries")
            return False

        # Get probe result from stub
        stub = self.storage.get_stub(doc_id)
        if not stub:
            return False

        probe_json = stub.get("probe_json", "")
        if not probe_json:
            return False

        probe_result = ProbeResult.model_validate_json(probe_json)

        # Generate summaries
        self.logger.info(f"Generating summaries for {doc_id}...")
        chunk_summaries = await self.librarian.summarize_chunks(probe_result)

        # Persist: update each chunk row with its summary, then delete+re-add
        summary_map: Dict[int, str] = {}
        for i, s in enumerate(chunk_summaries):
            if i < len(probe_result.chunks):
                summary_map[probe_result.chunks[i].index] = s

        for row in chunk_rows:
            idx = row.get("chunk_index", 0)
            row["summary"] = summary_map.get(idx, row.get("summary", ""))

        self.storage.update_chunks(doc_id, chunk_rows)
        self.logger.info(
            f"Persisted {len(chunk_summaries)} summaries for {doc_id}"
        )

        return True

    def stats(self) -> Dict[str, Any]:
        """Return system statistics."""
        return {
            "docs_indexed": self.storage.count_docs(),
            "chunks_stored": self.storage.count_chunks(),
            "cached_hashes": len(self.cache),
            "storage_size_bytes": self.file_store.total_size(),
            "collections": len(self.collections_store.list_collections()),
        }


class Collection:
    """A named group of documents — scoped view over shared storage.

    Acts as a thin wrapper around Mentat. Documents are indexed into the
    shared store; the collection just holds doc_id references.
    """

    def __init__(self, name: str, mentat: Mentat):
        self.name = name
        self._mentat = mentat
        self._store = mentat.collections_store

    @property
    def doc_ids(self) -> set:
        return self._store.get_doc_ids(self.name)

    async def add(
        self,
        path: str,
        force: bool = False,
        summarize: bool = False,
        use_llm_instructions: bool = False,
    ) -> str:
        """Index a file (if needed) and add it to this collection.

        Uses the shared cache — if the file was already indexed,
        just links the existing doc_id without re-processing.
        """
        doc_id = await self._mentat.add(
            path,
            force=force,
            summarize=summarize,
            use_llm_instructions=use_llm_instructions,
        )
        self._store.add_doc(self.name, doc_id)
        return doc_id

    async def search(
        self, query: str, top_k: int = 5, hybrid: bool = False
    ) -> List[MentatResult]:
        """Search only within this collection's documents."""
        ids = list(self.doc_ids)
        if not ids:
            return []
        return await self._mentat.search(
            query, top_k=top_k, hybrid=hybrid, doc_ids=ids
        )

    def remove(self, doc_id: str):
        """Remove a document from this collection (does NOT delete from storage)."""
        self._store.remove_doc(self.name, doc_id)

    def list_docs(self) -> List[Dict[str, Any]]:
        """List documents in this collection with their metadata."""
        docs = []
        for doc_id in self.doc_ids:
            stub = self._mentat.storage.get_stub(doc_id)
            if stub:
                docs.append(
                    {
                        "doc_id": doc_id,
                        "filename": stub.get("filename", ""),
                        "brief_intro": stub.get("brief_intro", ""),
                    }
                )
        return docs

    def delete(self) -> bool:
        """Delete this collection (does NOT delete underlying documents)."""
        return self._store.delete_collection(self.name)
