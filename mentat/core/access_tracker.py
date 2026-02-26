"""Two-layer FIFO queue for access-frequency-based processing triggers.

Tracks resource access patterns using two queues:
  - Layer 1 (recent): Fixed-size ordered dict of recently accessed keys.
  - Layer 2 (hot): Keys that were accessed again while still in Layer 1.

When a key is promoted from recent → hot, an optional async callback fires,
allowing consumers to trigger processing (e.g., embedding, summarization).
"""

import asyncio
import json
import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger("mentat.access_tracker")


class AccessTracker:
    """Two-layer FIFO queue for access-frequency-based processing triggers.

    Parameters:
        recent_size: Maximum number of entries in the recent (Layer 1) queue.
        hot_size: Maximum number of entries in the hot (Layer 2) queue.
        on_promote: Async callback fired when a key is promoted to hot queue.
            Receives the key as its only argument.
        persist_path: Optional file path for persisting heat map state as JSON.
            If provided, state is saved on changes (debounced) and loaded on init.
        save_debounce_seconds: Minimum interval between disk writes (default 5s).
    """

    def __init__(
        self,
        recent_size: int = 200,
        hot_size: int = 50,
        on_promote: Optional[Callable[[str], Awaitable[None]]] = None,
        persist_path: Optional[str] = None,
        save_debounce_seconds: float = 5.0,
    ):
        self._recent_size = max(1, recent_size)
        self._hot_size = max(1, hot_size)
        self._on_promote = on_promote
        self._persist_path = persist_path
        self._save_debounce = save_debounce_seconds
        self._save_handle: Optional[asyncio.TimerHandle] = None

        # Layer 1: recently accessed keys → timestamp of last access
        self._recent: OrderedDict[str, float] = OrderedDict()
        # Layer 2: hot keys (accessed ≥2 times while in recent) → timestamp promoted
        self._hot: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()

        # Load persisted state if available
        if self._persist_path:
            self._load()

    async def track(self, key: str) -> bool:
        """Record an access event for *key*.

        Returns ``True`` if *key* was promoted to the hot queue during this
        call, ``False`` otherwise.
        """
        async with self._lock:
            now = time.time()

            # Already hot — just refresh position
            if key in self._hot:
                self._hot.move_to_end(key)
                self._hot[key] = now
                self._schedule_save()
                return False

            # In recent queue — promote to hot
            if key in self._recent:
                del self._recent[key]
                self._hot[key] = now
                self._hot.move_to_end(key)
                # Evict oldest hot entry if over capacity
                while len(self._hot) > self._hot_size:
                    self._hot.popitem(last=False)
                logger.info(f"Promoted to hot queue: {key}")
                self._schedule_save()
                # Fire callback outside the lock
                if self._on_promote is not None:
                    asyncio.create_task(self._safe_promote(key))
                return True

            # New key — add to recent
            self._recent[key] = now
            self._recent.move_to_end(key)
            # Evict oldest recent entry if over capacity
            while len(self._recent) > self._recent_size:
                self._recent.popitem(last=False)
            self._schedule_save()
            return False

    def is_hot(self, key: str) -> bool:
        """Return ``True`` if *key* is in the hot queue."""
        return key in self._hot

    def is_recent(self, key: str) -> bool:
        """Return ``True`` if *key* is in the recent queue."""
        return key in self._recent

    def get_recent(self) -> List[str]:
        """Return keys in the recent queue (oldest first)."""
        return list(self._recent.keys())

    def get_hot(self) -> List[str]:
        """Return keys in the hot queue (oldest first)."""
        return list(self._hot.keys())

    def stats(self) -> Dict[str, Any]:
        """Return queue statistics."""
        return {
            "recent_count": len(self._recent),
            "recent_capacity": self._recent_size,
            "hot_count": len(self._hot),
            "hot_capacity": self._hot_size,
        }

    async def _safe_promote(self, key: str) -> None:
        """Fire the on_promote callback, swallowing errors."""
        try:
            await self._on_promote(key)  # type: ignore[misc]
        except Exception:
            logger.exception(f"on_promote callback failed for {key}")

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        """Load persisted heat map state from JSON file."""
        path = Path(self._persist_path)  # type: ignore[arg-type]
        if not path.exists():
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Heat map file corrupted, starting fresh: {path}")
            return

        for entry in data.get("recent", []):
            key = entry.get("key", "")
            ts = entry.get("timestamp", 0.0)
            if key:
                self._recent[key] = ts
        while len(self._recent) > self._recent_size:
            self._recent.popitem(last=False)

        for entry in data.get("hot", []):
            key = entry.get("key", "")
            ts = entry.get("timestamp", 0.0)
            if key:
                self._hot[key] = ts
        while len(self._hot) > self._hot_size:
            self._hot.popitem(last=False)

        logger.info(
            f"Loaded heat map: {len(self._recent)} recent, {len(self._hot)} hot"
        )

    def _save_to_disk(self) -> None:
        """Write current state to JSON file (synchronous)."""
        if not self._persist_path:
            return

        path = Path(self._persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "recent": [
                {"key": k, "timestamp": v} for k, v in self._recent.items()
            ],
            "hot": [
                {"key": k, "timestamp": v} for k, v in self._hot.items()
            ],
        }

        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except OSError:
            logger.warning(f"Failed to save heat map: {path}")

    def _schedule_save(self) -> None:
        """Schedule a debounced save. Multiple calls within the window coalesce."""
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
