import time
from contextlib import contextmanager
from typing import Dict, Optional
from pydantic import BaseModel


class TelemetryStats(BaseModel):
    probe_time_ms: float = 0.0
    librarian_time_ms: float = 0.0
    total_tokens: int = 0.0
    saved_context_ratio: float = 0.0


class Telemetry:
    _stats: Dict[str, TelemetryStats] = {}

    @classmethod
    @contextmanager
    def time_it(cls, doc_id: str, phase: str):
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        duration_ms = (end - start) * 1000

        if doc_id not in cls._stats:
            cls._stats[doc_id] = TelemetryStats()

        if phase == "probe":
            cls._stats[doc_id].probe_time_ms += duration_ms
        elif phase == "librarian":
            cls._stats[doc_id].librarian_time_ms += duration_ms

    @classmethod
    def record_tokens(cls, doc_id: str, tokens: int):
        if doc_id not in cls._stats:
            cls._stats[doc_id] = TelemetryStats()
        cls._stats[doc_id].total_tokens += tokens

    @classmethod
    def record_savings(cls, doc_id: str, ratio: float):
        if doc_id not in cls._stats:
            cls._stats[doc_id] = TelemetryStats()
        cls._stats[doc_id].saved_context_ratio = ratio

    @classmethod
    def get_stats(cls, doc_id: str) -> Optional[TelemetryStats]:
        return cls._stats.get(doc_id)

    @classmethod
    def format_stats(cls, doc_id: str) -> str:
        stats = cls.get_stats(doc_id)
        if not stats:
            return "No stats recorded."

        return (
            f"[Stats] Probed: {stats.probe_time_ms:.1f}ms | "
            f"Librarian: {stats.librarian_time_ms:.1f}ms | "
            f"Tokens: {stats.total_tokens} | "
            f"Saved: {stats.saved_context_ratio*100:.1f}% context"
        )
