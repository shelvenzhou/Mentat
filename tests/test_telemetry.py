import time

import pytest

from mentat.core.telemetry import Telemetry


@pytest.fixture(autouse=True)
def clear_telemetry():
    Telemetry._stats.clear()
    yield
    Telemetry._stats.clear()


def test_time_it_records_duration():
    with Telemetry.time_it("doc1", "probe"):
        time.sleep(0.01)

    stats = Telemetry.get_stats("doc1")
    assert stats is not None
    assert stats.probe_time_ms > 0


def test_record_tokens():
    Telemetry.record_tokens("doc1", 10)
    Telemetry.record_tokens("doc1", 5)

    stats = Telemetry.get_stats("doc1")
    assert stats is not None
    assert stats.total_tokens == 15


def test_record_savings():
    Telemetry.record_savings("doc1", 0.25)

    stats = Telemetry.get_stats("doc1")
    assert stats is not None
    assert stats.saved_context_ratio == 0.25


def test_format_stats():
    with Telemetry.time_it("doc1", "probe"):
        time.sleep(0.002)
    with Telemetry.time_it("doc1", "embedding"):
        time.sleep(0.002)

    Telemetry.record_tokens("doc1", 42)
    Telemetry.record_chunks("doc1", 3)
    Telemetry.record_savings("doc1", 0.5)

    out = Telemetry.format_stats("doc1")

    assert "[Stats] 3 chunks" in out
    assert "42 tokens" in out
    assert "Saved: 50.0% context" in out
    assert "Probe:" in out
    assert "Embedding:" in out
    assert "Total:" in out
