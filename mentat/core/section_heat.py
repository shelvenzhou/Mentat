"""Section-level heat tracking with weighted scoring and exponential time decay.

Tracks which document sections are accessed most frequently across search,
inspect, and read_segment operations.  Each access type carries a different
weight reflecting signal strength:

    read_segment  → 3.0  (explicit content request)
    inspect       → 2.0  (section-filtered inspection)
    search        → 1.0  (section appeared in search results)

Heat scores decay exponentially over time (configurable half-life, default 24h)
so recent accesses matter more than old ones.

Persistence follows the same debounced JSON pattern as AccessTracker.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger("mentat.section_heat")


@dataclass
class SectionHeatEntry:
    """A single section's accumulated heat score."""

    doc_id: str
    section: str
    raw_score: float = 0.0
    access_count: int = 0
    last_access: float = 0.0
    first_access: float = 0.0


class SectionHeatTracker:
    """Tracks section-level access importance with weighted scoring and time decay.

    Parameters:
        half_life_seconds: Exponential decay half-life (default 86400 = 24h).
        hot_threshold: Decayed score above which a section is considered "hot".
        max_entries: Maximum tracked entries (coldest evicted when exceeded).
        on_hot_section: Optional async callback(doc_id, section) when a section
            first crosses the hot threshold.  Reserved for future use.
        persist_path: Optional file path for JSON persistence.
        save_debounce_seconds: Minimum interval between disk writes (default 5s).
    """

    def __init__(
        self,
        half_life_seconds: float = 86400.0,
        hot_threshold: float = 5.0,
        max_entries: int = 1000,
        on_hot_section: Optional[Callable[[str, str], Awaitable[None]]] = None,
        persist_path: Optional[str] = None,
        save_debounce_seconds: float = 5.0,
    ):
        self._half_life = max(1.0, half_life_seconds)
        self._hot_threshold = hot_threshold
        self._max_entries = max(1, max_entries)
        self._on_hot_section = on_hot_section
        self._persist_path = persist_path
        self._save_debounce = save_debounce_seconds
        self._save_handle: Optional[asyncio.TimerHandle] = None

        # (doc_id, section) → SectionHeatEntry
        self._entries: Dict[Tuple[str, str], SectionHeatEntry] = {}
        # Track which keys have already fired the hot callback
        self._hot_fired: set = set()
        self._lock = asyncio.Lock()

        if self._persist_path:
            self._load()

    # ── Core API ──────────────────────────────────────────────────────

    async def record(
        self, doc_id: str, section: str, weight: float = 1.0
    ) -> bool:
        """Record an access event for a section.

        Returns True if the section crossed the hot threshold during this
        call (first time only), False otherwise.
        """
        async with self._lock:
            now = time.time()
            key = (doc_id, section)

            entry = self._entries.get(key)
            if entry is None:
                entry = SectionHeatEntry(
                    doc_id=doc_id,
                    section=section,
                    raw_score=weight,
                    access_count=1,
                    last_access=now,
                    first_access=now,
                )
                self._entries[key] = entry
            else:
                entry.raw_score += weight
                entry.access_count += 1
                entry.last_access = now

            # Evict coldest if over capacity
            if len(self._entries) > self._max_entries:
                self._evict_coldest()

            self._schedule_save()

            # Check hot threshold (fire callback only once per key)
            decayed = self._decayed_score(entry)
            if decayed >= self._hot_threshold and key not in self._hot_fired:
                self._hot_fired.add(key)
                logger.info(
                    "Section crossed hot threshold: %s :: %s (score=%.2f)",
                    doc_id[:8],
                    section,
                    decayed,
                )
                if self._on_hot_section is not None:
                    asyncio.create_task(
                        self._safe_hot_callback(doc_id, section)
                    )
                return True

            return False

    async def record_sections(
        self, doc_id: str, sections: Iterable[str], weight: float = 1.0
    ) -> None:
        """Record access for multiple sections of the same document."""
        for section in sections:
            section = section.strip()
            if section:
                await self.record(doc_id, section, weight=weight)

    # ── Query API ─────────────────────────────────────────────────────

    def get_score(self, doc_id: str, section: str) -> float:
        """Return the decayed score for a specific section."""
        entry = self._entries.get((doc_id, section))
        if entry is None:
            return 0.0
        return self._decayed_score(entry)

    def get_hot_sections(
        self, doc_id: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Return sections with decayed score above the hot threshold.

        Sorted by decayed score descending.  Optionally filtered by doc_id.
        """
        results = []
        for key, entry in self._entries.items():
            if doc_id is not None and entry.doc_id != doc_id:
                continue
            score = self._decayed_score(entry)
            if score >= self._hot_threshold:
                results.append(self._entry_to_dict(entry, score))

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def get_top_sections(
        self, doc_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Return top-N sections for a specific document by decayed score."""
        results = []
        for key, entry in self._entries.items():
            if entry.doc_id != doc_id:
                continue
            score = self._decayed_score(entry)
            if score > 0:
                results.append(self._entry_to_dict(entry, score))

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    def stats(self) -> Dict[str, Any]:
        """Return tracker statistics."""
        hot_count = sum(
            1
            for entry in self._entries.values()
            if self._decayed_score(entry) >= self._hot_threshold
        )
        return {
            "total_entries": len(self._entries),
            "hot_count": hot_count,
            "max_entries": self._max_entries,
            "half_life_seconds": self._half_life,
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _decayed_score(self, entry: SectionHeatEntry) -> float:
        """Compute decayed score: raw_score * 0.5^(elapsed / half_life)."""
        elapsed = time.time() - entry.last_access
        if elapsed <= 0:
            return entry.raw_score
        decay_factor = 0.5 ** (elapsed / self._half_life)
        return entry.raw_score * decay_factor

    @staticmethod
    def _entry_to_dict(entry: SectionHeatEntry, score: float) -> Dict[str, Any]:
        return {
            "doc_id": entry.doc_id,
            "section": entry.section,
            "score": round(score, 4),
            "raw_score": entry.raw_score,
            "access_count": entry.access_count,
            "last_access": entry.last_access,
        }

    def _evict_coldest(self) -> None:
        """Remove the entry with the lowest decayed score."""
        if not self._entries:
            return
        coldest_key = min(
            self._entries, key=lambda k: self._decayed_score(self._entries[k])
        )
        del self._entries[coldest_key]
        self._hot_fired.discard(coldest_key)

    async def _safe_hot_callback(self, doc_id: str, section: str) -> None:
        """Fire the on_hot_section callback, swallowing errors."""
        try:
            await self._on_hot_section(doc_id, section)  # type: ignore[misc]
        except Exception:
            logger.exception(
                "on_hot_section callback failed for %s :: %s", doc_id, section
            )

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted section heat map from JSON file."""
        path = Path(self._persist_path)  # type: ignore[arg-type]
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Section heat map file corrupted, starting fresh: %s", path)
            return

        for entry_data in data.get("entries", []):
            doc_id = entry_data.get("doc_id", "")
            section = entry_data.get("section", "")
            if not doc_id or not section:
                continue
            key = (doc_id, section)
            self._entries[key] = SectionHeatEntry(
                doc_id=doc_id,
                section=section,
                raw_score=entry_data.get("raw_score", 0.0),
                access_count=entry_data.get("access_count", 0),
                last_access=entry_data.get("last_access", 0.0),
                first_access=entry_data.get("first_access", 0.0),
            )

        # Trim to max_entries (evict coldest)
        while len(self._entries) > self._max_entries:
            self._evict_coldest()

        logger.info("Loaded section heat map: %d entries", len(self._entries))

    def _save_to_disk(self) -> None:
        """Write current state to JSON file (synchronous)."""
        if not self._persist_path:
            return

        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": 1,
            "entries": [
                {
                    "doc_id": e.doc_id,
                    "section": e.section,
                    "raw_score": e.raw_score,
                    "access_count": e.access_count,
                    "last_access": e.last_access,
                    "first_access": e.first_access,
                }
                for e in self._entries.values()
            ],
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            logger.warning("Failed to save section heat map: %s", path)

    def _schedule_save(self) -> None:
        """Schedule a debounced save.  Multiple calls within the window coalesce."""
        if not self._persist_path:
            return

        if self._save_handle is not None:
            self._save_handle.cancel()

        try:
            loop = asyncio.get_running_loop()
            self._save_handle = loop.call_later(
                self._save_debounce, self._save_to_disk
            )
        except RuntimeError:
            # No running loop (e.g., during shutdown) — save immediately
            self._save_to_disk()

    def save_now(self) -> None:
        """Force an immediate save (call during shutdown)."""
        if self._save_handle is not None:
            self._save_handle.cancel()
            self._save_handle = None
        self._save_to_disk()
