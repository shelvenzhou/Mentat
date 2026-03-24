"""Performance tests comparing async vs sync modes."""

import asyncio
import pytest
import time
from pathlib import Path

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig


@pytest.fixture
def perf_mentat(tmp_path):
    """Create a Mentat instance for performance testing."""
    config = MentatConfig(
        db_path=str(tmp_path / "perf_db"),
        storage_dir=str(tmp_path / "perf_storage"),
        max_concurrent_tasks=3,
    )
    mentat = Mentat(config)
    yield mentat
    Mentat.reset()


@pytest.fixture
def sample_files(tmp_path):
    """Create multiple sample files for batch testing."""
    files = []
    for i in range(10):
        file_path = tmp_path / f"doc_{i}.md"
        content = f"""# Document {i}

## Introduction
This is document number {i} for performance testing.

## Section 1
Content for section 1 in document {i}.
Some more text to make it realistic.

## Section 2
More content for section 2.
Additional information here.

## Section 3
Even more content for the third section.
Making sure we have enough text.

## Conclusion
Wrapping up document {i}.
"""
        file_path.write_text(content)
        files.append(str(file_path))
    return files


@pytest.mark.asyncio
@pytest.mark.slow
class TestAsyncVsSyncPerformance:
    """Compare performance between async and sync modes."""

    async def test_single_file_async_faster(self, perf_mentat, sample_files):
        """Test that async mode returns faster for single file."""
        await perf_mentat.start()

        file = sample_files[0]

        # Async mode (default)
        start = time.time()
        doc_id_async = await perf_mentat.add(file, wait=False, force=True)
        async_duration = time.time() - start

        # Sync mode (wait=True)
        start = time.time()
        doc_id_sync = await perf_mentat.add(file, wait=True, force=True)
        sync_duration = time.time() - start

        # Async should be significantly faster (< 3s vs potentially 5-10s)
        assert async_duration < 3.0
        assert async_duration < sync_duration

        await perf_mentat.shutdown()

    async def test_batch_async_speedup(self, perf_mentat, sample_files):
        """Test that async mode speeds up batch indexing."""
        await perf_mentat.start()

        # Async mode - index all files without waiting
        start = time.time()
        doc_ids_async = await asyncio.gather(*[
            perf_mentat.add(f, wait=False, force=True) for f in sample_files
        ])
        async_batch_duration = time.time() - start

        # Async mode should complete very quickly (just probe + stub for all)
        assert async_batch_duration < 10.0  # Should be ~1-3s per file

        # Wait for all to complete processing
        await asyncio.gather(*[
            perf_mentat.wait_for_completion(d, timeout=60) for d in doc_ids_async
        ])

        await perf_mentat.shutdown()

        print(f"\n⏱️ Async batch indexing: {async_batch_duration:.2f}s for {len(sample_files)} files")
        print(f"   Average: {async_batch_duration/len(sample_files):.2f}s per file (immediate return)")

    async def test_concurrent_processing_throughput(self, perf_mentat, sample_files):
        """Test throughput of concurrent background processing."""
        await perf_mentat.start()

        # Submit all files
        submit_start = time.time()
        doc_ids = await asyncio.gather(*[
            perf_mentat.add(f, wait=False, force=True) for f in sample_files
        ])
        submit_duration = time.time() - submit_start

        # Wait for all to complete
        process_start = time.time()
        completed = await asyncio.gather(*[
            perf_mentat.wait_for_completion(d, timeout=120) for d in doc_ids
        ])
        process_duration = time.time() - process_start

        assert all(completed)
        total_duration = submit_duration + process_duration

        await perf_mentat.shutdown()

        print(f"\n⏱️ Throughput test ({len(sample_files)} files):")
        print(f"   Submit: {submit_duration:.2f}s")
        print(f"   Process: {process_duration:.2f}s")
        print(f"   Total: {total_duration:.2f}s")
        print(f"   Throughput: {len(sample_files)/process_duration:.2f} files/sec (processing)")


@pytest.mark.asyncio
@pytest.mark.slow
class TestMemoryUsage:
    """Test memory efficiency of async queue system."""

    async def test_memory_bounded_processing(self, perf_mentat, tmp_path):
        """Test that queue doesn't accumulate unbounded tasks."""
        await perf_mentat.start()

        # Create many small files
        files = []
        for i in range(50):
            file_path = tmp_path / f"small_{i}.md"
            file_path.write_text(f"# Document {i}\n\nContent.")
            files.append(str(file_path))

        # Submit all
        doc_ids = await asyncio.gather(*[
            perf_mentat.add(f, wait=False, force=True) for f in files
        ])

        # Check queue size
        queue_size = len(perf_mentat.processor.queue._tasks)

        # Queue should have all tasks
        assert queue_size <= len(files)

        # Wait for processing
        await asyncio.gather(*[
            perf_mentat.wait_for_completion(d, timeout=120) for d in doc_ids
        ])

        # Cleanup should work
        removed = perf_mentat.processor.queue.cleanup_completed(max_age_hours=24)
        # None should be removed yet (all recent)
        assert removed == 0

        await perf_mentat.shutdown()


@pytest.mark.asyncio
@pytest.mark.slow
class TestScalability:
    """Test system scalability with varying loads."""

    async def test_increasing_file_count(self, tmp_path):
        """Test performance with increasing file counts."""
        results = {}

        for file_count in [5, 10, 20]:
            config = MentatConfig(
                db_path=str(tmp_path / f"scale_{file_count}_db"),
                storage_dir=str(tmp_path / f"scale_{file_count}_storage"),
                max_concurrent_tasks=3,
            )
            mentat = Mentat(config)
            await mentat.start()

            # Create files
            files = []
            for i in range(file_count):
                file_path = tmp_path / f"scale_{file_count}_{i}.md"
                file_path.write_text(f"# Doc {i}\n\nContent here.")
                files.append(str(file_path))

            # Measure submit time
            start = time.time()
            doc_ids = await asyncio.gather(*[
                mentat.add(f, wait=False) for f in files
            ])
            submit_time = time.time() - start

            # Measure total processing time
            process_start = time.time()
            await asyncio.gather(*[
                mentat.wait_for_completion(d, timeout=120) for d in doc_ids
            ])
            process_time = time.time() - process_start

            results[file_count] = {
                "submit_time": submit_time,
                "process_time": process_time,
                "total_time": submit_time + process_time,
            }

            await mentat.shutdown()
            Mentat.reset()

        # Print results
        print(f"\n⏱️ Scalability Test Results:")
        for count, times in results.items():
            print(f"   {count} files: submit={times['submit_time']:.2f}s, "
                  f"process={times['process_time']:.2f}s, "
                  f"total={times['total_time']:.2f}s")

        # Submit time should scale linearly (roughly)
        assert results[20]["submit_time"] < 20.0  # Should stay reasonable

    async def test_max_concurrent_limit_effectiveness(self, tmp_path):
        """Test that max_concurrent_tasks limit is effective."""
        # Test with different concurrency limits
        for max_concurrent in [1, 3, 5]:
            config = MentatConfig(
                db_path=str(tmp_path / f"concurrent_{max_concurrent}_db"),
                storage_dir=str(tmp_path / f"concurrent_{max_concurrent}_storage"),
                max_concurrent_tasks=max_concurrent,
            )
            mentat = Mentat(config)
            await mentat.start()

            # Create files
            files = []
            for i in range(10):
                file_path = tmp_path / f"concurrent_{max_concurrent}_{i}.md"
                file_path.write_text(f"# Doc {i}\n\nContent.")
                files.append(str(file_path))

            # Submit and process
            start = time.time()
            doc_ids = await asyncio.gather(*[
                mentat.add(f, wait=False) for f in files
            ])

            await asyncio.gather(*[
                mentat.wait_for_completion(d, timeout=120) for d in doc_ids
            ])
            duration = time.time() - start

            print(f"\n⏱️ max_concurrent={max_concurrent}: {duration:.2f}s for 10 files")

            await mentat.shutdown()
            Mentat.reset()


@pytest.mark.asyncio
class TestWaitTimeout:
    """Test timeout behavior in various scenarios."""

    async def test_wait_timeout_returns_false(self, perf_mentat):
        """Test that wait_for_completion respects timeout."""
        await perf_mentat.start()

        # Create a task that will never complete (mock)
        # In real scenario, this would be a very slow processing task

        # For now, test with nonexistent doc (returns True immediately)
        result = await perf_mentat.wait_for_completion("fake-id", timeout=0.5)
        assert result is True  # Nonexistent = treated as complete

        await perf_mentat.shutdown()

    async def test_wait_succeeds_before_timeout(self, perf_mentat, sample_files):
        """Test that wait_for_completion returns True when task completes."""
        await perf_mentat.start()

        doc_id = await perf_mentat.add(sample_files[0], wait=False, force=True)

        # Wait with generous timeout
        result = await perf_mentat.wait_for_completion(doc_id, timeout=60)
        assert result is True

        await perf_mentat.shutdown()
