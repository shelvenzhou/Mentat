"""Background processing queue for async document indexing.

This module provides an in-memory priority queue system for processing
document chunks (embeddings and summarization) in the background, allowing
immediate return from add() operations while enrichment happens asynchronously.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from mentat.probes.base import Chunk, ProbeResult

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.queue")


# ── Shared chunk preparation utilities ──────────────────────────────────────
# Used by both BackgroundProcessor (async path) and Mentat.add (inline path).

# Max tokens for embedding model (text-embedding-3-*: 8191)
# Use conservative limit since estimate_tokens() is approximate
_MAX_EMBED_TOKENS = 6000
_OVERLAP_CHARS = 500  # Overlap between split pieces for context continuity

# Type alias for chunk mapping entries
ChunkMapping = List[Tuple[int, int, int, int, int]]  # (chunk_idx, piece_idx, total_pieces, start_char, end_char)


def prepare_embed_texts(
    chunks: List[Chunk],
) -> Tuple[List[str], ChunkMapping]:
    """Build embedding-ready texts from probe chunks, splitting oversized ones.

    Returns:
        (embed_texts, chunk_mapping) — texts ready for embed_batch and a
        mapping from each text back to its source chunk.
    """
    from mentat.probes._utils import estimate_tokens

    embed_texts: List[str] = []
    chunk_mapping: ChunkMapping = []

    for i, chunk in enumerate(chunks):
        text = chunk.content
        section_prefix = f"[{chunk.section}] " if chunk.section else ""
        full_text = section_prefix + text
        estimated_tokens = estimate_tokens(full_text)

        if estimated_tokens <= _MAX_EMBED_TOKENS:
            embed_texts.append(full_text)
            chunk_mapping.append((i, 0, 1, 0, len(text)))
        else:
            max_chars = int(_MAX_EMBED_TOKENS * 3)
            pieces: List[Tuple[int, int]] = []
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                pieces.append((start, end))
                if end >= len(text):
                    break
                start = end - _OVERLAP_CHARS

            num_pieces = len(pieces)
            logger.debug(
                f"Splitting chunk {i} (~{estimated_tokens} tokens, {len(text)} chars) "
                f"into {num_pieces} pieces with {_OVERLAP_CHARS}ch overlap"
            )
            for piece_idx, (s, e) in enumerate(pieces):
                piece_marker = f" [part {piece_idx + 1}/{num_pieces}]"
                embed_texts.append(section_prefix + piece_marker + "\n" + text[s:e])
                chunk_mapping.append((i, piece_idx, num_pieces, s, e))

    return embed_texts, chunk_mapping


def _promote_metadata_fields(
    record: Dict[str, Any],
    extra_fields: Dict[str, Any],
    field_names: List[str],
) -> None:
    """Promote fields from metadata_json into the top-level record dict.

    Only promotes if the record's current value for the field is None
    (i.e. not already set by extra_fields directly).
    """
    promoted_any = False
    for name in field_names:
        if record.get(name) is not None:
            continue
        if not promoted_any:
            meta_json = extra_fields.get("metadata_json", "")
            if not meta_json or meta_json == "{}":
                return
            try:
                import json as _json
                meta = _json.loads(meta_json)
            except (ValueError, TypeError):
                return
            promoted_any = True
        if name in meta:
            record[name] = meta[name]


def build_chunk_records(
    doc_id: str,
    filename: str,
    chunks: List[Chunk],
    vectors: List[List[float]],
    summaries: List[str],
    chunk_mapping: ChunkMapping,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Build storage-ready chunk records from vectors + mapping.

    Args:
        extra_fields: Optional dict of fields to include in every record
            (e.g. source, indexed_at, file_type, metadata_json for filtering).

    Returns a list of dicts ready for ``LanceDBStorage.add_chunks``.
    """
    if len(vectors) != len(chunk_mapping):
        raise ValueError(
            f"Mismatched lengths: vectors={len(vectors)}, "
            f"chunk_mapping={len(chunk_mapping)}"
        )

    records: List[Dict[str, Any]] = []
    for vector, (chunk_idx, piece_idx, total_pieces, start_char, end_char) in zip(
        vectors, chunk_mapping
    ):
        chunk = chunks[chunk_idx]
        summary = summaries[chunk_idx] if chunk_idx < len(summaries) else ""

        if total_pieces == 1:
            content = chunk.content
            chunk_id = f"{doc_id}_{chunk.index}"
        else:
            content = chunk.content[start_char:end_char]
            chunk_id = f"{doc_id}_{chunk.index}_p{piece_idx}"

        record = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "filename": filename,
            "content": content,
            "summary": summary,
            "section": chunk.section or "",
            "chunk_index": chunk.index,
            "vector": vector,
            "is_split": total_pieces > 1,
            "piece_index": piece_idx if total_pieces > 1 else None,
            "total_pieces": total_pieces if total_pieces > 1 else None,
            "session_id": None,  # Always present; promoted from metadata below
        }
        if extra_fields:
            record.update(extra_fields)
            # Promote configured fields from metadata_json to top-level columns
            # for efficient LanceDB filtering.  Currently promotes: session_id.
            _promote_metadata_fields(record, extra_fields, ["session_id"])
        records.append(record)

    return records


@dataclass
class ProcessingTask:
    """A document processing task in the queue.

    Attributes:
        doc_id: Unique document identifier
        probe_result: Output from the probe layer (ToC, chunks, stats)
        priority: Higher values are processed first (default: 0)
        submitted_at: Timestamp when task was submitted
        status: Current processing state (pending/processing/completed/failed)
        error: Error message if processing failed
        needs_summarization: Whether to generate chunk summaries
    """

    doc_id: str
    probe_result: ProbeResult
    priority: int = 0
    submitted_at: float = field(default_factory=time.time)
    status: str = "pending"  # pending | processing | completed | failed
    error: Optional[str] = None
    needs_summarization: bool = False
    chunk_extra_fields: Optional[Dict[str, Any]] = None


class ProcessingQueue:
    """In-memory priority queue for processing tasks.

    Uses a heap list + asyncio.Event so that ``bump_priority`` actually
    affects ordering.  ``get_next`` always picks the highest-priority
    pending task (re-sorted after each bump).
    """

    def __init__(self):
        import heapq
        self._heap: List = []              # min-heap of (-priority, submit_order, doc_id)
        self._counter: int = 0             # tie-breaker for equal priorities (FIFO)
        self._tasks: Dict[str, ProcessingTask] = {}
        self._lock = asyncio.Lock()
        self._event = asyncio.Event()      # signalled when work is available

    async def submit(self, task: ProcessingTask) -> None:
        """Submit a task to the processing queue.

        If a task with the same doc_id already exists and is pending/processing,
        the submission is skipped to prevent duplicate processing.
        """
        import heapq

        async with self._lock:
            if task.doc_id in self._tasks:
                existing = self._tasks[task.doc_id]
                if existing.status in ("pending", "processing"):
                    logger.warning(
                        f"Task {task.doc_id} already in queue with status "
                        f"{existing.status}, skipping duplicate submission"
                    )
                    return

            self._tasks[task.doc_id] = task
            heapq.heappush(self._heap, (-task.priority, self._counter, task.doc_id))
            self._counter += 1
            logger.debug(
                f"Submitted task {task.doc_id} with priority {task.priority} "
                f"(summarize={task.needs_summarization})"
            )
        self._event.set()

    def get_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a processing task."""
        task = self._tasks.get(doc_id)
        if not task:
            return None

        return {
            "doc_id": doc_id,
            "status": task.status,
            "submitted_at": task.submitted_at,
            "error": task.error,
            "needs_summarization": task.needs_summarization,
        }

    def bump_priority(self, doc_id: str, delta: int = 10) -> None:
        """Increase priority for a queued document and re-sort the heap.

        Args:
            doc_id: Document identifier
            delta: Priority increase amount (default: 10)
        """
        import heapq

        task = self._tasks.get(doc_id)
        if not task or task.status != "pending":
            return

        task.priority += delta

        # Rebuild heap entries for this doc_id with updated priority.
        # We add a new entry with the bumped priority; stale entries are
        # skipped in get_next() when the task is no longer pending.
        heapq.heappush(self._heap, (-task.priority, self._counter, doc_id))
        self._counter += 1
        logger.debug(f"Bumped priority for {doc_id} to {task.priority}")

    def cleanup_completed(self, max_age_hours: float = 24) -> int:
        """Remove completed/failed tasks older than max_age."""
        cutoff = time.time() - (max_age_hours * 3600)

        to_remove = [
            doc_id
            for doc_id, task in self._tasks.items()
            if task.status in ("completed", "failed") and task.submitted_at < cutoff
        ]

        for doc_id in to_remove:
            del self._tasks[doc_id]

        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} old tasks")

        return len(to_remove)

    async def get_next(self) -> Optional[str]:
        """Get next pending task from the heap.

        Skips stale entries (tasks already processing/completed/failed).
        Returns None if nothing is available after a 1-second wait.
        """
        import heapq

        # Try to pop from heap first (no wait needed if work is ready)
        async with self._lock:
            while self._heap:
                neg_prio, _order, doc_id = heapq.heappop(self._heap)
                task = self._tasks.get(doc_id)
                if task and task.status == "pending":
                    return doc_id
                # Stale entry — skip and continue

        # Heap was empty — wait for a submit signal (with timeout for cleanup)
        self._event.clear()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            return None

        # Re-check after wakeup
        async with self._lock:
            while self._heap:
                neg_prio, _order, doc_id = heapq.heappop(self._heap)
                task = self._tasks.get(doc_id)
                if task and task.status == "pending":
                    return doc_id
            return None


class BackgroundProcessor:
    """Background worker for processing document chunks.

    Runs a worker loop that:
    1. Pulls tasks from the priority queue
    2. Generates embeddings for all chunks
    3. Optionally generates summaries for chunks
    4. Stores chunks with vectors and summaries in the database

    Attributes:
        mentat: Parent Mentat instance (provides storage, librarian, embeddings)
        queue: ProcessingQueue instance
        max_concurrent: Maximum number of documents to process simultaneously
    """

    def __init__(self, mentat: "Mentat", max_concurrent: int = 3):
        self.mentat = mentat
        self.queue = ProcessingQueue()
        self.max_concurrent = max_concurrent
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Start the background processing worker."""
        if self._running:
            logger.warning("Background processor already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(self._process_loop())
        logger.info(
            f"Started background processor (max_concurrent={self.max_concurrent})"
        )

    async def stop(self) -> None:
        """Stop the background processor gracefully.

        Waits for currently processing tasks to complete before shutting down.
        """
        if not self._running:
            return

        self._running = False
        # Wake get_next() if it's blocked waiting for work.
        self.queue._event.set()
        if self._worker_task:
            await self._worker_task
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)

        logger.info("Background processor stopped")

    def _has_unfinished_tasks(self) -> bool:
        """Return True if queue still has pending/processing work."""
        return any(
            task.status in ("pending", "processing")
            for task in self.queue._tasks.values()
        )

    async def _process_loop(self) -> None:
        """Main worker loop - runs continuously while _running is True."""
        cleanup_interval = 3600  # Clean up old tasks every hour
        last_cleanup = time.time()

        while self._running or self._has_unfinished_tasks():
            try:
                # Get next task (with timeout to allow periodic cleanup)
                doc_id = await self.queue.get_next()

                if doc_id:
                    task = self.queue._tasks.get(doc_id)
                    if task:
                        # Process with concurrency limit
                        processing_task = asyncio.create_task(
                            self._process_with_semaphore(task)
                        )
                        self._active_tasks.add(processing_task)
                        processing_task.add_done_callback(self._active_tasks.discard)

                # Periodic cleanup
                if time.time() - last_cleanup > cleanup_interval:
                    self.queue.cleanup_completed()
                    last_cleanup = time.time()

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)

    async def _process_with_semaphore(self, task: ProcessingTask) -> None:
        """Process a task with concurrency control.

        Args:
            task: ProcessingTask to process
        """
        async with self._semaphore:
            await self._process_task(task)

    async def _process_task(self, task: ProcessingTask) -> None:
        """Process a single document task.

        Args:
            task: ProcessingTask to process
        """
        task.status = "processing"
        logger.info(
            f"Processing task {task.doc_id} (chunks={len(task.probe_result.chunks)}, "
            f"summarize={task.needs_summarization})"
        )

        start_time = time.time()

        try:
            # Run embedding + summarization concurrently
            embed_coro = self._embed_chunks(task)
            summ_coro = (
                self._summarize_chunks(task)
                if task.needs_summarization
                else self._noop_summaries(task)
            )

            vectors, summaries = await asyncio.gather(
                embed_coro, summ_coro, return_exceptions=True
            )

            # Check for errors
            if isinstance(vectors, Exception):
                raise vectors
            if isinstance(summaries, Exception):
                raise summaries

            # Store chunks with vectors + summaries
            await self._store_chunks(task, vectors, summaries)

            # Mark as completed
            task.status = "completed"
            duration = time.time() - start_time
            logger.info(
                f"Completed task {task.doc_id} in {duration:.2f}s "
                f"({len(vectors)} chunks)"
            )

        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            logger.error(
                f"Processing failed for {task.doc_id}: {e}",
                exc_info=True
            )

    async def _embed_chunks(self, task: ProcessingTask) -> List[List[float]]:
        """Generate embeddings for all chunks.

        Splits oversized chunks into multiple pieces to fit within the embedding
        model's context window. Each piece gets its own embedding vector and will
        be stored as a separate searchable chunk.

        Args:
            task: ProcessingTask with probe_result containing chunks

        Returns:
            List of embedding vectors (may be more than original chunks if splitting occurred)
        """
        chunks = task.probe_result.chunks
        if not chunks:
            return []

        embed_texts, chunk_mapping = prepare_embed_texts(chunks)

        logger.debug(
            f"Embedding {len(embed_texts)} pieces from {len(chunks)} semantic chunks for {task.doc_id}"
        )
        vectors = await self.mentat.embeddings.embed_batch(embed_texts)

        # Store mapping for use in _store_chunks
        task.chunk_mapping = chunk_mapping

        return vectors

    async def _summarize_chunks(self, task: ProcessingTask) -> List[str]:
        """Generate summaries for all chunks.

        Args:
            task: ProcessingTask with probe_result containing chunks

        Returns:
            List of chunk summaries (one per chunk)
        """
        logger.debug(f"Summarizing chunks for {task.doc_id}")
        summaries = await self.mentat.librarian.summarize_chunks(task.probe_result)
        return summaries

    async def _noop_summaries(self, task: ProcessingTask) -> List[str]:
        """Return empty summaries (fast mode - no summarization).

        Args:
            task: ProcessingTask with probe_result containing chunks

        Returns:
            List of empty strings (one per chunk)
        """
        return [""] * len(task.probe_result.chunks)

    async def _store_chunks(
        self, task: ProcessingTask, vectors: List[List[float]], summaries: List[str]
    ) -> None:
        """Store chunks with vectors and summaries in the database."""
        chunks = task.probe_result.chunks
        doc_id = task.doc_id
        filename = task.probe_result.filename or "unknown"
        chunk_mapping = getattr(task, 'chunk_mapping', None)

        if not chunk_mapping:
            # Fallback: build identity mapping (no splits)
            chunk_mapping = [(i, 0, 1, 0, len(c.content)) for i, c in enumerate(chunks)]

        chunk_records = build_chunk_records(
            doc_id, filename, chunks, vectors, summaries, chunk_mapping,
            extra_fields=task.chunk_extra_fields,
        )

        logger.debug(
            f"Storing {len(chunk_records)} chunks "
            f"(from {len(chunks)} semantic chunks) for {doc_id}"
        )

        vector_dim = len(vectors[0]) if vectors else 0
        self.mentat.storage._ensure_chunks_table(vector_dim)
        self.mentat.storage.add_chunks(chunk_records)
