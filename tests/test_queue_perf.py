import asyncio
import time
from unittest.mock import AsyncMock, Mock

import pytest

from mentat.core.queue import BackgroundProcessor, ProcessingTask
from mentat.probes.base import Chunk, ProbeResult, StructureInfo, TopicInfo


def _probe_result() -> ProbeResult:
    return ProbeResult(
        file_type="markdown",
        filename="perf.md",
        topic=TopicInfo(title="Perf"),
        structure=StructureInfo(toc=[]),
        chunks=[Chunk(content="content " * 40, index=0, section="A")],
        stats={"is_full_content": False},
    )


def _mock_mentat(process_delay: float = 0.0):
    m = Mock()

    async def _embed(texts):
        if process_delay:
            await asyncio.sleep(process_delay)
        return [[0.1, 0.2, 0.3] for _ in texts]

    m.embeddings = Mock()
    m.embeddings.embed_batch = AsyncMock(side_effect=_embed)
    m.librarian = Mock()
    m.librarian.summarize_chunks = AsyncMock(return_value=["summary"])
    m.storage = Mock()
    m.storage._ensure_chunks_table = Mock()
    m.storage.add_chunks = Mock()
    return m


async def _wait_all_completed(processor: BackgroundProcessor, doc_ids, timeout: float = 20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        done = True
        for doc_id in doc_ids:
            status = processor.queue.get_status(doc_id)
            if not status or status["status"] not in {"completed", "failed"}:
                done = False
                break
        if done:
            return True
        await asyncio.sleep(0.02)
    return False


@pytest.mark.asyncio
async def test_queue_throughput_100_tasks():
    processor = BackgroundProcessor(_mock_mentat(process_delay=0.005), max_concurrent=5)
    await processor.start()

    tasks = [
        ProcessingTask(doc_id=f"doc-{i}", probe_result=_probe_result())
        for i in range(100)
    ]

    t0 = time.time()
    for task in tasks:
        await processor.queue.submit(task)

    ok = await _wait_all_completed(processor, [t.doc_id for t in tasks], timeout=20)
    duration = time.time() - t0
    throughput = 100 / max(duration, 1e-6)

    await processor.stop()

    assert ok is True
    assert throughput > 20


@pytest.mark.asyncio
async def test_queue_latency_submit_to_start():
    processor = BackgroundProcessor(_mock_mentat(process_delay=0.05), max_concurrent=1)
    await processor.start()

    task = ProcessingTask(doc_id="latency-doc", probe_result=_probe_result())
    t0 = time.time()
    await processor.queue.submit(task)

    started = False
    while time.time() - t0 < 2.0:
        status = processor.queue.get_status("latency-doc")
        if status and status["status"] == "processing":
            started = True
            break
        await asyncio.sleep(0.01)

    await processor.stop()

    assert started is True
    assert (time.time() - t0) < 2.0


@pytest.mark.asyncio
async def test_queue_memory_bounded():
    processor = BackgroundProcessor(_mock_mentat(process_delay=0.0), max_concurrent=5)
    await processor.start()

    tasks = [
        ProcessingTask(doc_id=f"mem-{i}", probe_result=_probe_result())
        for i in range(120)
    ]
    for task in tasks:
        await processor.queue.submit(task)

    ok = await _wait_all_completed(processor, [t.doc_id for t in tasks], timeout=20)
    removed = processor.queue.cleanup_completed(max_age_hours=0)

    await processor.stop()

    assert ok is True
    assert removed == 120
    assert len(processor.queue._tasks) == 0


@pytest.mark.asyncio
async def test_concurrency_scaling():
    async def _run(max_concurrent: int) -> float:
        processor = BackgroundProcessor(
            _mock_mentat(process_delay=0.02), max_concurrent=max_concurrent
        )
        await processor.start()

        tasks = [
            ProcessingTask(doc_id=f"c{max_concurrent}-{i}", probe_result=_probe_result())
            for i in range(30)
        ]

        t0 = time.time()
        for task in tasks:
            await processor.queue.submit(task)
        ok = await _wait_all_completed(processor, [t.doc_id for t in tasks], timeout=30)
        duration = time.time() - t0

        await processor.stop()
        assert ok is True
        return duration

    t1 = await _run(1)
    t3 = await _run(3)
    t5 = await _run(5)

    assert t1 > t3
    assert t3 >= t5 or t3 > (t5 * 0.8)
