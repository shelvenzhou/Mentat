"""Background processing queue for async document indexing.

This module provides an in-memory priority queue system for processing
document chunks (embeddings and summarization) in the background, allowing
immediate return from add() operations while enrichment happens asynchronously.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mentat.probes.base import Chunk, ProbeResult

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.queue")


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


class ProcessingQueue:
    """In-memory priority queue for processing tasks.

    Manages a priority queue of documents awaiting embedding and summarization.
    Tracks task status in memory (lost on restart).
    """

    def __init__(self):
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._tasks: Dict[str, ProcessingTask] = {}
        self._lock = asyncio.Lock()

    async def submit(self, task: ProcessingTask) -> None:
        """Submit a task to the processing queue.

        Args:
            task: ProcessingTask to queue

        Note:
            If a task with the same doc_id already exists and is pending/processing,
            the submission is skipped to prevent duplicate processing.
        """
        async with self._lock:
            # Prevent duplicate submissions
            if task.doc_id in self._tasks:
                existing = self._tasks[task.doc_id]
                if existing.status in ("pending", "processing"):
                    logger.warning(
                        f"Task {task.doc_id} already in queue with status "
                        f"{existing.status}, skipping duplicate submission"
                    )
                    return

            self._tasks[task.doc_id] = task
            # Negative priority for max-heap behavior (higher priority = processed first)
            await self._queue.put((-task.priority, task.doc_id))
            logger.debug(
                f"Submitted task {task.doc_id} with priority {task.priority} "
                f"(summarize={task.needs_summarization})"
            )

    def get_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a processing task.

        Args:
            doc_id: Document identifier

        Returns:
            Status dict with keys: status, submitted_at, error (if failed)
            Returns None if doc_id not found in queue.
        """
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
        """Increase priority for a queued document.

        Useful for prioritizing documents that are queried before processing
        completes.

        Args:
            doc_id: Document identifier
            delta: Priority increase amount (default: 10)

        Note:
            Priority change only affects future queue ordering. Tasks already
            pulled from the queue are unaffected.
        """
        task = self._tasks.get(doc_id)
        if task and task.status == "pending":
            task.priority += delta
            logger.debug(f"Bumped priority for {doc_id} to {task.priority}")

    def cleanup_completed(self, max_age_hours: float = 24) -> int:
        """Remove completed/failed tasks older than max_age.

        Args:
            max_age_hours: Maximum age in hours for completed tasks

        Returns:
            Number of tasks removed
        """
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
        """Get next task from queue (blocking).

        Returns:
            doc_id of next task, or None if queue is empty after timeout
        """
        try:
            _, doc_id = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            return doc_id
        except asyncio.TimeoutError:
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
        if self._worker_task:
            await self._worker_task

        logger.info("Background processor stopped")

    async def _process_loop(self) -> None:
        """Main worker loop - runs continuously while _running is True."""
        cleanup_interval = 3600  # Clean up old tasks every hour
        last_cleanup = time.time()

        while self._running:
            try:
                # Get next task (with timeout to allow periodic cleanup)
                doc_id = await self.queue.get_next()

                if doc_id:
                    task = self.queue._tasks.get(doc_id)
                    if task:
                        # Process with concurrency limit
                        asyncio.create_task(self._process_with_semaphore(task))

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
        from mentat.probes._utils import estimate_tokens

        chunks = task.probe_result.chunks

        if not chunks:
            return []

        # Max tokens for embedding model (text-embedding-3-small: 8191)
        # Use very conservative limit since estimate_tokens() is approximate
        MAX_EMBED_TOKENS = 6000
        OVERLAP_CHARS = 500  # Overlap between split pieces for context continuity

        # Track mapping from embed pieces back to original chunks
        embed_texts = []
        chunk_mapping = []  # List of (chunk_index, piece_index, total_pieces, start_char, end_char)

        for i, chunk in enumerate(chunks):
            text = chunk.content
            section_prefix = f"[{chunk.section}] " if chunk.section else ""

            # Check if chunk needs splitting
            full_text = section_prefix + text
            estimated_tokens = estimate_tokens(full_text)

            if estimated_tokens <= MAX_EMBED_TOKENS:
                # Chunk fits - embed as-is
                embed_texts.append(full_text)
                chunk_mapping.append((i, 0, 1, 0, len(text)))
            else:
                # Chunk too large - split into overlapping pieces
                max_chars = int(MAX_EMBED_TOKENS * 3.5)
                pieces = []
                start = 0

                while start < len(text):
                    end = min(start + max_chars, len(text))
                    pieces.append((start, end))
                    if end >= len(text):
                        break
                    start = end - OVERLAP_CHARS  # Overlap for context

                num_pieces = len(pieces)
                logger.debug(
                    f"Splitting chunk {i} (~{estimated_tokens} tokens, {len(text)} chars) "
                    f"into {num_pieces} pieces with {OVERLAP_CHARS}ch overlap"
                )

                for piece_idx, (start, end) in enumerate(pieces):
                    piece_text = text[start:end]
                    piece_marker = f" [part {piece_idx + 1}/{num_pieces}]"
                    embed_text = section_prefix + piece_marker + "\n" + piece_text

                    embed_texts.append(embed_text)
                    chunk_mapping.append((i, piece_idx, num_pieces, start, end))

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
        """Store chunks with vectors and summaries in the database.

        Handles split chunks: if a semantic chunk was split into multiple pieces
        for embedding, each piece is stored as a separate searchable chunk with
        metadata indicating it's part of a larger whole.

        Args:
            task: ProcessingTask being processed
            vectors: List of embedding vectors (may be > len(chunks) if splitting occurred)
            summaries: List of chunk summaries (one per original semantic chunk)
        """
        chunks = task.probe_result.chunks
        doc_id = task.doc_id
        filename = task.probe_result.filename or "unknown"

        # Check if we have chunk mapping (from split chunks)
        chunk_mapping = getattr(task, 'chunk_mapping', None)

        if chunk_mapping:
            # We have split chunks - use mapping to build records
            if len(vectors) != len(chunk_mapping):
                raise ValueError(
                    f"Mismatched lengths: vectors={len(vectors)}, "
                    f"chunk_mapping={len(chunk_mapping)}"
                )

            chunk_records = []
            for vector, (chunk_idx, piece_idx, total_pieces, start_char, end_char) in zip(
                vectors, chunk_mapping
            ):
                chunk = chunks[chunk_idx]
                summary = summaries[chunk_idx] if chunk_idx < len(summaries) else ""

                # Extract the piece content
                if total_pieces == 1:
                    # Not split
                    content = chunk.content
                    chunk_id = f"{doc_id}_{chunk.index}"
                else:
                    # Split piece
                    content = chunk.content[start_char:end_char]
                    chunk_id = f"{doc_id}_{chunk.index}_p{piece_idx}"

                chunk_records.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "content": content,
                    "summary": summary,  # Same summary for all pieces
                    "section": chunk.section or "",
                    "chunk_index": chunk.index,
                    "vector": vector,
                    # Metadata for split pieces
                    "is_split": total_pieces > 1,
                    "piece_index": piece_idx if total_pieces > 1 else None,
                    "total_pieces": total_pieces if total_pieces > 1 else None,
                })

            logger.debug(
                f"Stored {len(chunk_records)} searchable chunks "
                f"(from {len(chunks)} semantic chunks) for {doc_id}"
            )
        else:
            # No splitting - original behavior
            if len(vectors) != len(chunks) or len(summaries) != len(chunks):
                raise ValueError(
                    f"Mismatched lengths: chunks={len(chunks)}, "
                    f"vectors={len(vectors)}, summaries={len(summaries)}"
                )

            chunk_records = []
            for chunk, summary, vector in zip(chunks, summaries, vectors):
                chunk_records.append({
                    "chunk_id": f"{doc_id}_{chunk.index}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "content": chunk.content,
                    "summary": summary,
                    "section": chunk.section or "",
                    "chunk_index": chunk.index,
                    "vector": vector,
                })

            logger.debug(f"Stored {len(chunk_records)} chunks for {doc_id}")

        # Store in database (this ensures chunks table is created with proper vector dim)
        vector_dim = len(vectors[0]) if vectors else 0
        self.mentat.storage._ensure_chunks_table(vector_dim)
        self.mentat.storage.add_chunks(chunk_records)
