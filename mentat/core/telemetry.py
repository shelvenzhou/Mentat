import time
from contextlib import contextmanager
from typing import Dict, Optional
from pydantic import BaseModel


def _fmt_time(ms: float) -> str:
    """Format milliseconds into a human-readable string."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


class TelemetryStats(BaseModel):
    probe_time_ms: float = 0.0
    summarize_time_ms: float = 0.0
    librarian_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    total_tokens: int = 0
    saved_context_ratio: float = 0.0
    num_chunks: int = 0
    fast_mode: bool = False  # True if template instructions + no summarization


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
        elif phase == "summarize":
            cls._stats[doc_id].summarize_time_ms += duration_ms
        elif phase == "librarian":
            cls._stats[doc_id].librarian_time_ms += duration_ms
        elif phase == "embedding":
            cls._stats[doc_id].embedding_time_ms += duration_ms

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
    def record_chunks(cls, doc_id: str, num_chunks: int):
        if doc_id not in cls._stats:
            cls._stats[doc_id] = TelemetryStats()
        cls._stats[doc_id].num_chunks = num_chunks

    @classmethod
    def get_stats(cls, doc_id: str) -> Optional[TelemetryStats]:
        return cls._stats.get(doc_id)

    @classmethod
    def format_stats(cls, doc_id: str) -> str:
        stats = cls.get_stats(doc_id)
        if not stats:
            return "No stats recorded."

        total = (
            stats.probe_time_ms
            + stats.summarize_time_ms
            + stats.librarian_time_ms
            + stats.embedding_time_ms
        )

        mode_indicator = " | ⚡ Fast mode" if stats.fast_mode else ""

        lines = [
            f"[Stats] {stats.num_chunks} chunks | "
            f"{stats.total_tokens if stats.total_tokens else 0} tokens | "
            f"Saved: {stats.saved_context_ratio * 100:.1f}% context{mode_indicator}",
            f"  Probe:       {_fmt_time(stats.probe_time_ms)}",
        ]

        # Only show summarize/instruction times if they were actually used
        if stats.summarize_time_ms > 0:
            lines.append(f"  Summarize:   {_fmt_time(stats.summarize_time_ms)}")
        if stats.librarian_time_ms > 0:
            lines.append(
                f"  Instruction: {_fmt_time(stats.librarian_time_ms)}"
                + (" (template)" if stats.fast_mode else " (LLM)")
            )

        lines.append(f"  Embedding:   {_fmt_time(stats.embedding_time_ms)}")
        lines.append(f"  Total:       {_fmt_time(total)}")

        return "\n".join(lines)
