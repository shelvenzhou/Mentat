import asyncio

import pytest

from mentat.core.access_tracker import AccessTracker


@pytest.mark.asyncio
async def test_track_new_key_goes_to_recent():
    tracker = AccessTracker(recent_size=5, hot_size=5)
    promoted = await tracker.track("k1")

    assert promoted is False
    assert tracker.is_recent("k1") is True
    assert tracker.is_hot("k1") is False
    assert tracker.get_recent() == ["k1"]


@pytest.mark.asyncio
async def test_track_twice_promotes_to_hot():
    tracker = AccessTracker(recent_size=5, hot_size=5)
    await tracker.track("k1")
    promoted = await tracker.track("k1")

    assert promoted is True
    assert tracker.is_recent("k1") is False
    assert tracker.is_hot("k1") is True


@pytest.mark.asyncio
async def test_on_promote_callback_fires():
    seen = []
    event = asyncio.Event()

    async def on_promote(key: str):
        seen.append(key)
        event.set()

    tracker = AccessTracker(recent_size=5, hot_size=5, on_promote=on_promote)

    await tracker.track("k1")
    await tracker.track("k1")
    await asyncio.wait_for(event.wait(), timeout=1.0)

    assert seen == ["k1"]


@pytest.mark.asyncio
async def test_recent_fifo_eviction():
    tracker = AccessTracker(recent_size=2, hot_size=5)
    await tracker.track("k1")
    await tracker.track("k2")
    await tracker.track("k3")

    assert tracker.get_recent() == ["k2", "k3"]
    assert tracker.is_recent("k1") is False


@pytest.mark.asyncio
async def test_hot_fifo_eviction():
    tracker = AccessTracker(recent_size=10, hot_size=2)

    for key in ["k1", "k2", "k3"]:
        await tracker.track(key)
        await tracker.track(key)

    assert tracker.get_hot() == ["k2", "k3"]
    assert tracker.is_hot("k1") is False


@pytest.mark.asyncio
async def test_is_hot_is_recent():
    tracker = AccessTracker(recent_size=5, hot_size=5)

    await tracker.track("k1")
    assert tracker.is_recent("k1") is True
    assert tracker.is_hot("k1") is False

    await tracker.track("k1")
    assert tracker.is_recent("k1") is False
    assert tracker.is_hot("k1") is True


def test_stats():
    tracker = AccessTracker(recent_size=3, hot_size=2)
    stats = tracker.stats()

    assert stats["recent_count"] == 0
    assert stats["recent_capacity"] == 3
    assert stats["hot_count"] == 0
    assert stats["hot_capacity"] == 2


# ── Persistence Tests ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persistence_save_and_load(tmp_path):
    path = str(tmp_path / "heat_map.json")

    tracker1 = AccessTracker(recent_size=5, hot_size=5, persist_path=path)
    await tracker1.track("k1")
    await tracker1.track("k2")
    await tracker1.track("k2")  # promote k2 to hot
    tracker1.save_now()

    # Create new tracker from same file
    tracker2 = AccessTracker(recent_size=5, hot_size=5, persist_path=path)
    assert tracker2.is_recent("k1")
    assert tracker2.is_hot("k2")
    assert not tracker2.is_recent("k2")


@pytest.mark.asyncio
async def test_persistence_missing_file():
    tracker = AccessTracker(
        recent_size=5, hot_size=5, persist_path="/tmp/nonexistent_dir_xyz/heat.json"
    )
    # Should not raise, just start empty
    assert tracker.stats()["recent_count"] == 0


def test_save_now_no_persist_path():
    tracker = AccessTracker(recent_size=5, hot_size=5)
    tracker.save_now()  # Should be a no-op, no error


@pytest.mark.asyncio
async def test_persistence_corrupted_file(tmp_path):
    path = tmp_path / "heat_map.json"
    path.write_text("not valid json{{{")

    tracker = AccessTracker(recent_size=5, hot_size=5, persist_path=str(path))
    # Should recover gracefully, start empty
    assert tracker.stats()["recent_count"] == 0
    assert tracker.stats()["hot_count"] == 0
