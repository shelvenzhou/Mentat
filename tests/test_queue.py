"""Tests for the async processing queue system."""

import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from mentat.core.queue import ProcessingTask, ProcessingQueue, BackgroundProcessor
from mentat.probes.base import ProbeResult, TopicInfo, StructureInfo, Chunk


@pytest.fixture
def sample_probe_result():
    """Create a sample ProbeResult for testing."""
    return ProbeResult(
        file_type="markdown",
        filename="test.md",
        topic=TopicInfo(title="Test Document"),
        structure=StructureInfo(toc=[]),
        chunks=[
            Chunk(content="First chunk", index=0, section="Introduction"),
            Chunk(content="Second chunk", index=1, section="Body"),
            Chunk(content="Third chunk", index=2, section="Conclusion"),
        ],
        stats={"is_full_content": False},
    )


@pytest.fixture
def processing_task(sample_probe_result):
    """Create a sample ProcessingTask."""
    return ProcessingTask(
        doc_id="test-doc-123",
        probe_result=sample_probe_result,
        priority=0,
        needs_summarization=False,
    )


class TestProcessingTask:
    """Tests for ProcessingTask dataclass."""

    def test_create_task(self, sample_probe_result):
        task = ProcessingTask(
            doc_id="test-123",
            probe_result=sample_probe_result,
            priority=5,
            needs_summarization=True,
        )
        assert task.doc_id == "test-123"
        assert task.priority == 5
        assert task.status == "pending"
        assert task.needs_summarization is True
        assert task.error is None

    def test_default_values(self, sample_probe_result):
        task = ProcessingTask(
            doc_id="test-123",
            probe_result=sample_probe_result,
        )
        assert task.priority == 0
        assert task.status == "pending"
        assert task.needs_summarization is False

    def test_submitted_at_timestamp(self, sample_probe_result):
        before = time.time()
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)
        after = time.time()
        assert before <= task.submitted_at <= after


class TestProcessingQueue:
    """Tests for ProcessingQueue class."""

    @pytest.mark.asyncio
    async def test_submit_task(self, processing_task):
        queue = ProcessingQueue()
        await queue.submit(processing_task)

        status = queue.get_status("test-doc-123")
        assert status is not None
        assert status["status"] == "pending"
        assert status["doc_id"] == "test-doc-123"

    @pytest.mark.asyncio
    async def test_submit_duplicate_task_skipped(self, processing_task):
        queue = ProcessingQueue()
        await queue.submit(processing_task)

        # Try to submit the same doc_id again
        duplicate_task = ProcessingTask(
            doc_id="test-doc-123",
            probe_result=processing_task.probe_result,
            priority=10,  # Different priority
        )
        await queue.submit(duplicate_task)

        # Should still have the original task (not updated)
        assert len(queue._tasks) == 1
        assert queue._tasks["test-doc-123"].priority == 0  # Original priority

    @pytest.mark.asyncio
    async def test_priority_ordering(self, sample_probe_result):
        queue = ProcessingQueue()

        # Submit tasks with different priorities
        low_priority = ProcessingTask(
            doc_id="low", probe_result=sample_probe_result, priority=1
        )
        high_priority = ProcessingTask(
            doc_id="high", probe_result=sample_probe_result, priority=10
        )
        medium_priority = ProcessingTask(
            doc_id="medium", probe_result=sample_probe_result, priority=5
        )

        await queue.submit(low_priority)
        await queue.submit(high_priority)
        await queue.submit(medium_priority)

        # Get tasks in priority order (high to low)
        first = await queue.get_next()
        second = await queue.get_next()
        third = await queue.get_next()

        assert first == "high"
        assert second == "medium"
        assert third == "low"

    def test_get_status_not_found(self):
        queue = ProcessingQueue()
        status = queue.get_status("nonexistent")
        assert status is None

    def test_bump_priority(self, processing_task):
        queue = ProcessingQueue()
        asyncio.run(queue.submit(processing_task))

        # Bump priority
        queue.bump_priority("test-doc-123", delta=15)

        task = queue._tasks["test-doc-123"]
        assert task.priority == 15  # 0 + 15

    def test_bump_priority_nonexistent_task(self):
        queue = ProcessingQueue()
        # Should not raise error
        queue.bump_priority("nonexistent", delta=10)

    def test_cleanup_completed(self, sample_probe_result):
        queue = ProcessingQueue()

        # Create old completed task
        old_task = ProcessingTask(doc_id="old", probe_result=sample_probe_result)
        old_task.status = "completed"
        old_task.submitted_at = time.time() - (25 * 3600)  # 25 hours ago
        queue._tasks["old"] = old_task

        # Create recent completed task
        recent_task = ProcessingTask(doc_id="recent", probe_result=sample_probe_result)
        recent_task.status = "completed"
        recent_task.submitted_at = time.time() - 3600  # 1 hour ago
        queue._tasks["recent"] = recent_task

        # Create pending task
        pending_task = ProcessingTask(doc_id="pending", probe_result=sample_probe_result)
        queue._tasks["pending"] = pending_task

        # Cleanup with 24 hour threshold
        removed = queue.cleanup_completed(max_age_hours=24)

        assert removed == 1
        assert "old" not in queue._tasks
        assert "recent" in queue._tasks
        assert "pending" in queue._tasks

    @pytest.mark.asyncio
    async def test_get_next_timeout(self):
        queue = ProcessingQueue()
        # Should timeout and return None since queue is empty
        result = await queue.get_next()
        assert result is None

    @pytest.mark.asyncio
    async def test_queue_concurrent_priority_bump(self, sample_probe_result):
        queue = ProcessingQueue()
        await queue.submit(
            ProcessingTask(doc_id="prio-doc", probe_result=sample_probe_result, priority=0)
        )

        async def bump_many(n: int):
            for _ in range(n):
                queue.bump_priority("prio-doc", delta=1)
                await asyncio.sleep(0)

        workers = 10
        per_worker = 100
        await asyncio.gather(*(bump_many(per_worker) for _ in range(workers)))

        assert queue._tasks["prio-doc"].priority == workers * per_worker

    @pytest.mark.asyncio
    async def test_queue_equal_priority_fifo(self, sample_probe_result):
        queue = ProcessingQueue()

        await queue.submit(ProcessingTask(doc_id="a", probe_result=sample_probe_result, priority=5))
        await queue.submit(ProcessingTask(doc_id="b", probe_result=sample_probe_result, priority=5))
        await queue.submit(ProcessingTask(doc_id="c", probe_result=sample_probe_result, priority=5))

        assert await queue.get_next() == "a"
        assert await queue.get_next() == "b"
        assert await queue.get_next() == "c"


class TestBackgroundProcessor:
    """Tests for BackgroundProcessor class."""

    @pytest.fixture
    def mock_mentat(self, tmp_path):
        """Create a mock Mentat instance."""
        mentat = Mock()
        mentat.config.db_path = str(tmp_path / "db")
        mentat.embeddings = Mock()
        mentat.embeddings.embed_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mentat.librarian = Mock()
        mentat.librarian.summarize_chunks = AsyncMock(return_value=["summary1", "summary2", "summary3"])
        mentat.storage = Mock()
        mentat.storage._ensure_chunks_table = Mock()
        mentat.storage.add_chunks = Mock()
        return mentat

    @pytest.mark.asyncio
    async def test_start_and_stop(self, mock_mentat):
        processor = BackgroundProcessor(mock_mentat)
        await processor.start()

        assert processor._running is True
        assert processor._worker_task is not None

        await processor.stop()
        assert processor._running is False

    @pytest.mark.asyncio
    async def test_embed_chunks(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)

        # Mock embedding response (3 chunks = 3 vectors)
        mock_mentat.embeddings.embed_batch = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        vectors = await processor._embed_chunks(task)

        assert len(vectors) == 3
        assert vectors[0] == [0.1, 0.2]
        # Verify embed_batch was called with section-prefixed content
        call_args = mock_mentat.embeddings.embed_batch.call_args[0][0]
        assert "[Introduction] First chunk" in call_args[0]

    @pytest.mark.asyncio
    async def test_summarize_chunks(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)

        summaries = await processor._summarize_chunks(task)

        assert len(summaries) == 3
        mock_mentat.librarian.summarize_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_noop_summaries(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)

        summaries = await processor._noop_summaries(task)

        assert summaries == ["", "", ""]
        mock_mentat.librarian.summarize_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_chunks(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test-123", probe_result=sample_probe_result)

        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        summaries = ["sum1", "sum2", "sum3"]

        await processor._store_chunks(task, vectors, summaries)

        # Verify storage calls
        mock_mentat.storage._ensure_chunks_table.assert_called_once_with(2)  # vector_dim = 2
        mock_mentat.storage.add_chunks.assert_called_once()

        # Verify chunk records structure
        call_args = mock_mentat.storage.add_chunks.call_args[0][0]
        assert len(call_args) == 3
        assert call_args[0]["chunk_id"] == "test-123_0"
        assert call_args[0]["doc_id"] == "test-123"
        assert call_args[0]["content"] == "First chunk"
        assert call_args[0]["summary"] == "sum1"
        assert call_args[0]["vector"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_store_chunks_mismatched_lengths(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)

        vectors = [[0.1, 0.2]]  # Only 1 vector but 3 chunks
        summaries = ["sum1", "sum2", "sum3"]

        with pytest.raises(ValueError, match="Mismatched lengths"):
            await processor._store_chunks(task, vectors, summaries)

    @pytest.mark.asyncio
    async def test_process_task_success(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat, max_concurrent=1)
        task = ProcessingTask(
            doc_id="test", probe_result=sample_probe_result, needs_summarization=True
        )

        # Mock responses
        mock_mentat.embeddings.embed_batch = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )
        mock_mentat.librarian.summarize_chunks = AsyncMock(
            return_value=["sum1", "sum2", "sum3"]
        )

        await processor._process_task(task)

        assert task.status == "completed"
        assert task.error is None

    @pytest.mark.asyncio
    async def test_process_task_failure(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(doc_id="test", probe_result=sample_probe_result)

        # Mock embedding to raise error
        mock_mentat.embeddings.embed_batch = AsyncMock(side_effect=Exception("API Error"))

        await processor._process_task(task)

        assert task.status == "failed"
        assert "API Error" in task.error

    @pytest.mark.asyncio
    async def test_process_task_without_summarization(self, mock_mentat, sample_probe_result):
        processor = BackgroundProcessor(mock_mentat)
        task = ProcessingTask(
            doc_id="test", probe_result=sample_probe_result, needs_summarization=False
        )

        mock_mentat.embeddings.embed_batch = AsyncMock(
            return_value=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        )

        await processor._process_task(task)

        assert task.status == "completed"
        # Verify summarization was not called
        mock_mentat.librarian.summarize_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, mock_mentat, sample_probe_result):
        """Test that max_concurrent limits parallel processing."""
        processor = BackgroundProcessor(mock_mentat, max_concurrent=2)

        # Create tasks
        tasks = [
            ProcessingTask(doc_id=f"doc-{i}", probe_result=sample_probe_result)
            for i in range(5)
        ]

        # Mock slow processing
        async def slow_embed(*args, **kwargs):
            await asyncio.sleep(0.1)
            return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_mentat.embeddings.embed_batch = AsyncMock(side_effect=slow_embed)

        # Process tasks with semaphore limit
        processing_coros = [processor._process_with_semaphore(task) for task in tasks]

        start = time.time()
        await asyncio.gather(*processing_coros)
        duration = time.time() - start

        # With max_concurrent=2 and 5 tasks at 0.1s each:
        # Expected time: ~0.3s (3 batches: 2+2+1)
        # If no limit: ~0.1s (all parallel)
        assert duration > 0.2  # Definitely limited
        assert all(task.status == "completed" for task in tasks)

    @pytest.mark.asyncio
    async def test_queue_graceful_shutdown_during_processing(self, mock_mentat, sample_probe_result):
        """Shutdown during processing should still finish queued tasks."""
        processor = BackgroundProcessor(mock_mentat, max_concurrent=1)

        async def slow_embed(*args, **kwargs):
            await asyncio.sleep(0.05)
            return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_mentat.embeddings.embed_batch = AsyncMock(side_effect=slow_embed)

        await processor.start()
        doc_ids = [f"doc-{i}" for i in range(3)]
        for doc_id in doc_ids:
            await processor.queue.submit(
                ProcessingTask(doc_id=doc_id, probe_result=sample_probe_result)
            )

        # Give worker a moment to start processing, then stop.
        await asyncio.sleep(0.02)
        await processor.stop()

        statuses = [processor.queue.get_status(doc_id) for doc_id in doc_ids]
        assert all(s and s["status"] == "completed" for s in statuses)

    @pytest.mark.asyncio
    async def test_queue_task_error_recovery(self, mock_mentat, sample_probe_result):
        """Failure of one task should not block subsequent tasks."""
        processor = BackgroundProcessor(mock_mentat, max_concurrent=1)

        call_count = 0

        async def flaky_embed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("embedding failure")
            return [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_mentat.embeddings.embed_batch = AsyncMock(side_effect=flaky_embed)

        await processor.start()
        await processor.queue.submit(
            ProcessingTask(doc_id="fail-doc", probe_result=sample_probe_result)
        )
        await processor.queue.submit(
            ProcessingTask(doc_id="ok-doc", probe_result=sample_probe_result)
        )

        timeout = time.time() + 5
        while time.time() < timeout:
            s1 = processor.queue.get_status("fail-doc")
            s2 = processor.queue.get_status("ok-doc")
            if (
                s1 and s2
                and s1["status"] in ("failed", "completed")
                and s2["status"] in ("failed", "completed")
            ):
                break
            await asyncio.sleep(0.05)

        await processor.stop()

        assert processor.queue.get_status("fail-doc")["status"] == "failed"
        assert processor.queue.get_status("ok-doc")["status"] == "completed"


class TestIntegration:
    """Integration tests for the full queue workflow."""

    @pytest.fixture
    def mock_mentat_full(self, tmp_path):
        """Create a full mock Mentat instance with storage."""
        mentat = Mock()
        mentat.config.db_path = str(tmp_path / "db")
        mentat.embeddings = Mock()
        mentat.embeddings.embed_batch = AsyncMock(
            return_value=[[0.1, 0.2, 0.3] for _ in range(3)]
        )
        mentat.librarian = Mock()
        mentat.librarian.summarize_chunks = AsyncMock(
            return_value=["Summary 1", "Summary 2", "Summary 3"]
        )
        mentat.storage = Mock()
        mentat.storage._ensure_chunks_table = Mock()
        mentat.storage.add_chunks = Mock()
        return mentat

    @pytest.mark.asyncio
    async def test_full_workflow(self, mock_mentat_full, sample_probe_result):
        """Test complete workflow: submit → process → store."""
        processor = BackgroundProcessor(mock_mentat_full, max_concurrent=1)
        await processor.start()

        # Submit task
        task = ProcessingTask(
            doc_id="test-doc",
            probe_result=sample_probe_result,
            needs_summarization=True,
        )
        await processor.queue.submit(task)

        # Wait for processing (with timeout)
        timeout = 5.0
        start = time.time()
        while time.time() - start < timeout:
            status = processor.queue.get_status("test-doc")
            if status and status["status"] in ("completed", "failed"):
                break
            await asyncio.sleep(0.1)

        await processor.stop()

        # Verify processing completed
        status = processor.queue.get_status("test-doc")
        assert status["status"] == "completed"

        # Verify storage was called
        mock_mentat_full.storage.add_chunks.assert_called_once()

    @pytest.mark.asyncio
    async def test_priority_boost_on_pending_task(self, mock_mentat_full, sample_probe_result):
        """Test that pending tasks can have their priority boosted."""
        processor = BackgroundProcessor(mock_mentat_full)
        await processor.start()

        # Submit low priority task
        task = ProcessingTask(
            doc_id="test-doc",
            probe_result=sample_probe_result,
            priority=0,
        )
        await processor.queue.submit(task)

        # Boost priority (simulating search query)
        processor.queue.bump_priority("test-doc", delta=20)

        # Verify priority was increased
        updated_task = processor.queue._tasks["test-doc"]
        assert updated_task.priority == 20

        await processor.stop()

    @pytest.mark.asyncio
    async def test_multiple_tasks_with_cleanup(self, mock_mentat_full, sample_probe_result):
        """Test processing multiple tasks and cleanup."""
        processor = BackgroundProcessor(mock_mentat_full, max_concurrent=2)
        await processor.start()

        # Submit multiple tasks
        for i in range(5):
            task = ProcessingTask(
                doc_id=f"doc-{i}",
                probe_result=sample_probe_result,
            )
            await processor.queue.submit(task)

        # Wait for all to complete
        timeout = 10.0
        start = time.time()
        while time.time() - start < timeout:
            statuses = [
                processor.queue.get_status(f"doc-{i}")
                for i in range(5)
            ]
            if all(s and s["status"] in ("completed", "failed") for s in statuses):
                break
            await asyncio.sleep(0.1)

        # Cleanup old tasks (none should be cleaned since they're recent)
        removed = processor.queue.cleanup_completed(max_age_hours=24)
        assert removed == 0

        # All tasks should still be in queue
        assert len(processor.queue._tasks) == 5

        await processor.stop()
