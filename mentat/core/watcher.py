"""Per-collection file watcher with content-hash dedup and throttling.

Reads ``watch_paths`` / ``watch_ignore`` from ``CollectionStore`` and
maintains one async watch task per collection.  When a file changes,
it is re-indexed (only if its SHA-256 hash actually changed) and added
to the owning collection.

Uses ``watchfiles`` (Rust backend) for efficient filesystem monitoring.
"""

import asyncio
import fnmatch
import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.watcher")

# Minimum seconds between re-processing the same file path
_THROTTLE_SECONDS = 5.0


def _sha256(path: str) -> Optional[str]:
    """Return hex SHA-256 of a file, or None if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _make_filter(ignore_patterns: list[str]) -> Callable[..., bool]:
    """Build a watchfiles filter function from glob ignore patterns."""
    def _filter(change: Any, path: str) -> bool:
        name = Path(path).name
        for pattern in ignore_patterns:
            if fnmatch.fnmatch(name, pattern):
                return False
            # Also match against relative path segments (e.g. "node_modules")
            if pattern in Path(path).parts:
                return False
        return True
    return _filter


class MentatWatcher:
    """Watch directories for file changes, auto-reindex on content change.

    Each collection with ``watch_paths`` gets its own async watch task.
    File changes are indexed globally; the doc_id is added to the
    owning collection automatically.
    """

    def __init__(self, mentat: "Mentat"):
        self._mentat = mentat
        self._tasks: Dict[str, asyncio.Task] = {}  # collection_name -> task
        self._throttle: Dict[str, float] = {}  # abs_path -> last_process_time
        self._hashes: Dict[str, str] = {}  # abs_path -> last known SHA-256
        self._running = False

    async def start(self):
        """Start watching all collections that have watch_paths."""
        self._running = True
        await self.sync()
        logger.info("Watcher started")

    async def stop(self):
        """Stop all watch tasks."""
        self._running = False
        for name, task in self._tasks.items():
            task.cancel()
        for name, task in self._tasks.items():
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        logger.info("Watcher stopped")

    async def sync(self):
        """Sync watch tasks with current collection watch configs.

        Call after creating/updating/deleting collections to pick up changes.
        """
        configs = self._mentat.collections_store.get_all_watch_configs()

        # Stop tasks for collections that no longer have watch_paths
        stale = set(self._tasks.keys()) - set(configs.keys())
        for name in stale:
            self._tasks[name].cancel()
            try:
                await self._tasks[name]
            except (asyncio.CancelledError, Exception):
                pass
            del self._tasks[name]
            logger.info("Stopped watcher for collection %r (removed)", name)

        # Start/restart tasks for collections with watch_paths
        for name, config in configs.items():
            paths = config.get("watch_paths", [])
            ignore = config.get("watch_ignore", [])
            if not paths:
                continue
            # If already running with same config, skip
            if name in self._tasks and not self._tasks[name].done():
                continue
            self._tasks[name] = asyncio.create_task(
                self._watch_collection(name, paths, ignore),
                name=f"watcher:{name}",
            )
            logger.info("Started watcher for collection %r: %s", name, paths)

    def _matches_ignore(self, ignore: list[str], path: Path) -> bool:
        """Check if a path matches any ignore pattern."""
        name = path.name
        for pattern in ignore:
            if fnmatch.fnmatch(name, pattern):
                return True
            if pattern in path.parts:
                return True
        return False

    async def _initial_scan(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Index existing files in watched directories (skip already-indexed via cache)."""
        count = 0
        for dir_path in paths:
            for file_path in Path(dir_path).rglob("*"):
                if not file_path.is_file():
                    continue
                if self._matches_ignore(ignore, file_path):
                    continue
                path_str = str(file_path)
                try:
                    doc_id = await self._mentat.add(
                        path_str,
                        force=False,
                        source=f"watcher:{collection_name}",
                        collection=collection_name,
                    )
                    # Seed hash so _handle_change skips unchanged files
                    h = _sha256(path_str)
                    if h:
                        self._hashes[path_str] = h
                    count += 1
                except Exception:
                    logger.exception("Initial scan failed for %s", path_str)
        if count:
            logger.info(
                "Initial scan for %r: processed %d file(s)", collection_name, count
            )

    async def _watch_collection(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Watch a set of directories and index changes into a collection."""
        from watchfiles import awatch, Change

        # Resolve and filter to existing directories
        resolved_paths: list[str] = []
        for p in paths:
            expanded = str(Path(p).expanduser().resolve())
            if Path(expanded).is_dir():
                resolved_paths.append(expanded)
            else:
                logger.warning("Watch path %r is not a directory, skipping", p)

        if not resolved_paths:
            logger.warning("No valid watch paths for collection %r", collection_name)
            return

        # Index existing files before starting live watch
        await self._initial_scan(collection_name, resolved_paths, ignore)

        watch_filter = _make_filter(ignore)

        try:
            async for changes in awatch(
                *resolved_paths,
                watch_filter=watch_filter,
                stop_event=asyncio.Event() if not self._running else None,
            ):
                if not self._running:
                    break
                for change_type, path_str in changes:
                    if change_type == Change.deleted:
                        continue
                    if not Path(path_str).is_file():
                        continue
                    await self._handle_change(collection_name, path_str)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Watcher for %r crashed", collection_name)

    async def _handle_change(self, collection_name: str, path: str):
        """Process a single file change: throttle, hash-check, re-index."""
        now = time.monotonic()

        # Throttle: skip if recently processed
        last = self._throttle.get(path, 0.0)
        if now - last < _THROTTLE_SECONDS:
            return
        self._throttle[path] = now

        # Content hash check: skip if unchanged
        new_hash = _sha256(path)
        if new_hash is None:
            return
        old_hash = self._hashes.get(path)
        if new_hash == old_hash:
            return
        self._hashes[path] = new_hash

        # Re-index with force=True (content changed)
        try:
            doc_id = await self._mentat.add(
                path,
                force=True,
                source=f"watcher:{collection_name}",
                collection=collection_name,
            )
            logger.info(
                "Re-indexed %s -> %s (collection: %s)",
                Path(path).name, doc_id, collection_name,
            )
        except Exception:
            logger.exception("Failed to re-index %s", path)
