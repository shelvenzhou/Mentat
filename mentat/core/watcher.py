"""Per-collection file watcher with content-hash dedup and throttling.

Reads ``watch_paths`` / ``watch_ignore`` from ``CollectionStore`` and
maintains one async watch task per collection.  When a file changes,
it is re-indexed (only if its SHA-256 hash actually changed) and added
to the owning collection.

Supports two modes per collection:

- **full** (default): re-index entire file on content change (hash-based dedup).
- **append**: track byte offset, index only newly appended content as
  separate documents.  Designed for append-only files like JSONL logs.

Uses ``watchfiles`` (Rust backend) for efficient filesystem monitoring.
"""

import asyncio
import fnmatch
import hashlib
import json as json_module
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

logger = logging.getLogger("mentat.watcher")

# Minimum seconds between re-processing the same file path
_THROTTLE_SECONDS = 5.0

_OFFSETS_FILENAME = "watcher_offsets.json"


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
        self._offsets: Dict[str, int] = {}  # abs_path -> last processed byte offset
        self._configs: Dict[str, Dict[str, Any]] = {}  # collection_name -> watch config
        self._running = False

    # ── Offset persistence ──────────────────────────────────────────

    def _offsets_path(self) -> Path:
        return Path(self._mentat.config.db_path) / _OFFSETS_FILENAME

    def _load_offsets(self):
        p = self._offsets_path()
        if p.exists():
            try:
                self._offsets = json_module.loads(p.read_text())
                logger.info("Loaded %d watcher offsets from %s", len(self._offsets), p)
            except Exception:
                logger.warning("Failed to load watcher offsets, starting fresh")
                self._offsets = {}

    def _save_offsets(self):
        if not self._offsets:
            return
        p = self._offsets_path()
        tmp = p.with_suffix(".tmp")
        try:
            tmp.write_text(json_module.dumps(self._offsets))
            import os
            os.replace(str(tmp), str(p))  # atomic on POSIX
        except Exception:
            logger.exception("Failed to save watcher offsets")
            tmp.unlink(missing_ok=True)

    # ── Lifecycle ───────────────────────────────────────────────────

    async def start(self):
        """Start watching all collections that have watch_paths."""
        self._running = True
        self._load_offsets()
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
        self._save_offsets()
        logger.info("Watcher stopped")

    async def sync(self):
        """Sync watch tasks with current collection watch configs.

        Call after creating/updating/deleting collections to pick up changes.
        """
        configs = self._mentat.collections_store.get_all_watch_configs()
        self._configs = configs

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

    def _is_append_mode(self, collection_name: str) -> bool:
        config = self._configs.get(collection_name, {})
        return config.get("watch_mode") == "append"

    def _get_probe_config(self, collection_name: str) -> Optional[Dict[str, Any]]:
        config = self._configs.get(collection_name, {})
        return config.get("watch_probe_config")

    # ── Initial scan ────────────────────────────────────────────────

    async def _initial_scan(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Index existing files in watched directories."""
        is_append = self._is_append_mode(collection_name)

        if is_append:
            await self._initial_scan_append(collection_name, paths, ignore)
        else:
            await self._initial_scan_full(collection_name, paths, ignore)

    async def _initial_scan_full(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Full mode initial scan (skip already-indexed via cache)."""
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

    async def _initial_scan_append(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Append mode initial scan: process unindexed portions of files.

        Files with persisted offsets resume from where they left off.
        Files without offsets are processed from byte 0.

        Also verifies that docs from previous offsets actually have chunks
        in the vector DB.  If they don't (e.g. embedding failed), the
        offset is rolled back so the content is re-indexed.
        """
        config = self._configs.get(collection_name, {})
        recent_days = config.get("initial_scan_recent_days")
        now_ts = time.time()

        count = 0
        deferred: List[str] = []

        for dir_path in paths:
            for file_path in Path(dir_path).rglob("*"):
                if not file_path.is_file():
                    continue
                if self._matches_ignore(ignore, file_path):
                    continue
                path_str = str(file_path)

                # If we have a persisted offset, verify prior chunks exist
                if path_str in self._offsets:
                    self._verify_offset_integrity(collection_name, path_str)
                    await self._handle_append_change(collection_name, path_str)
                    count += 1
                    continue

                # New file — check age for prioritization
                if recent_days is not None:
                    try:
                        mtime = file_path.stat().st_mtime
                        age_days = (now_ts - mtime) / 86400
                        if age_days > recent_days:
                            deferred.append(path_str)
                            continue
                    except OSError:
                        continue

                await self._handle_append_change(collection_name, path_str)
                count += 1

        if count:
            logger.info(
                "Append initial scan for %r: processed %d file(s)", collection_name, count
            )

        # Deferred (older) files: process in background
        if deferred:
            logger.info(
                "Append initial scan for %r: deferring %d older file(s) to background",
                collection_name, len(deferred),
            )
            asyncio.create_task(
                self._process_deferred(collection_name, deferred),
                name=f"watcher:deferred:{collection_name}",
            )

    async def _process_deferred(self, collection_name: str, paths: list[str]):
        """Background task to index older files one at a time."""
        for path_str in paths:
            if not self._running:
                break
            try:
                await self._handle_append_change(collection_name, path_str)
            except Exception:
                logger.exception("Deferred scan failed for %s", path_str)
            # Yield to event loop between files
            await asyncio.sleep(0.1)

    # ── Watch loop ──────────────────────────────────────────────────

    async def _watch_collection(
        self, collection_name: str, paths: list[str], ignore: list[str]
    ):
        """Watch a set of directories and index changes into a collection."""
        from watchfiles import awatch, Change

        # Resolve paths, waiting for directories that don't exist yet
        resolved_paths: list[str] = []
        for p in paths:
            expanded = str(Path(p).expanduser().resolve())
            if Path(expanded).is_dir():
                resolved_paths.append(expanded)
            else:
                logger.info("Watch path %r does not exist yet, waiting...", p)
                while self._running and not Path(expanded).is_dir():
                    await asyncio.sleep(2)
                if Path(expanded).is_dir():
                    resolved_paths.append(expanded)
                    logger.info("Watch path %r appeared", p)

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
                        if self._is_append_mode(collection_name):
                            self._offsets.pop(path_str, None)
                            # Purge vecdb chunks for this session
                            session_id = Path(path_str).stem
                            removed = self._mentat.storage.delete_docs_by_session_id(session_id)
                            for doc_id in removed:
                                self._mentat.cache.remove(doc_id)
                                self._mentat.path_index.remove(doc_id)
                                self._mentat.collections_store.remove_doc(collection_name, doc_id)
                            if removed:
                                logger.info(
                                    "Purged %d docs for deleted session %s (collection: %s)",
                                    len(removed), session_id, collection_name,
                                )
                        continue
                    if not Path(path_str).is_file():
                        continue
                    await self._handle_change(collection_name, path_str)
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Watcher for %r crashed", collection_name)

    # ── Offset integrity ──────────────────────────────────────────────

    def _verify_offset_integrity(self, collection_name: str, path: str):
        """Check that prior append-indexed segments actually have chunks.

        If the vector DB is empty (cold start after DB deletion) but
        ``watcher_offsets.json`` survived, or if prior embedding failed,
        the offset is stale.  Roll it back to 0 so the full file is
        re-indexed on the next ``_handle_append_change`` call.
        """
        storage = self._mentat.storage
        coll_store = self._mentat.collections_store
        coll = coll_store.get(collection_name)
        if coll is None:
            return

        doc_ids = coll.get("doc_ids", [])
        if not doc_ids:
            # Collection has no docs at all — offsets are certainly stale
            old_offset = self._offsets.get(path, 0)
            if old_offset > 0:
                logger.warning(
                    "Offset integrity: collection %r is empty but offset for %s is %d — resetting to 0",
                    collection_name, Path(path).name, old_offset,
                )
                self._offsets[path] = 0
            return

        # Check whether any doc from this file actually has chunks.
        # Watcher-created docs use logical filenames like "session.jsonl@{offset}".
        stem = Path(path).name
        has_any_chunks = False
        for doc_id in doc_ids:
            stub = storage.get_stub(doc_id)
            if stub and stub.get("filename", "").startswith(stem):
                if storage.has_chunks(doc_id):
                    has_any_chunks = True
                    break

        if not has_any_chunks:
            old_offset = self._offsets.get(path, 0)
            if old_offset > 0:
                logger.warning(
                    "Offset integrity: no chunks found for %s in collection %r — resetting offset from %d to 0",
                    Path(path).name, collection_name, old_offset,
                )
                self._offsets[path] = 0

    # ── Change handlers ─────────────────────────────────────────────

    async def _handle_change(self, collection_name: str, path: str):
        """Route to full or append handler based on collection config."""
        if self._is_append_mode(collection_name):
            await self._handle_append_change(collection_name, path)
        else:
            await self._handle_full_change(collection_name, path)

    async def _handle_full_change(self, collection_name: str, path: str):
        """Full mode: throttle, hash-check, re-index entire file."""
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

    async def _handle_append_change(self, collection_name: str, path: str):
        """Append mode: read only new bytes, create new document per batch."""
        now = time.monotonic()

        # Throttle
        last = self._throttle.get(path, 0.0)
        if now - last < _THROTTLE_SECONDS:
            return
        self._throttle[path] = now

        # Check file size
        try:
            file_size = Path(path).stat().st_size
        except FileNotFoundError:
            self._offsets.pop(path, None)
            return

        last_offset = self._offsets.get(path, 0)
        if file_size <= last_offset:
            return  # No growth (possibly truncated/rewritten)

        # Read only new bytes
        try:
            with open(path, "rb") as f:
                f.seek(last_offset)
                new_bytes = f.read(file_size - last_offset)
        except OSError:
            logger.exception("Failed to read new bytes from %s", path)
            return

        # Discard incomplete last line
        last_newline = new_bytes.rfind(b"\n")
        if last_newline == -1:
            return  # No complete line yet
        new_bytes = new_bytes[:last_newline + 1]
        actual_end = last_offset + last_newline + 1

        if not new_bytes.strip():
            self._offsets[path] = actual_end
            return

        # Write to temp file for probing
        storage_dir = Path(self._mentat.config.storage_dir)
        stem = Path(path).stem
        tmp_path = storage_dir / f"_append_{stem}_{last_offset}.jsonl"
        try:
            tmp_path.write_bytes(new_bytes)
        except OSError:
            logger.exception("Failed to write temp file %s", tmp_path)
            return

        try:
            probe_config = self._get_probe_config(collection_name)
            session_id = stem  # filename stem as session identifier
            doc_id = await self._mentat.add(
                str(tmp_path),
                force=True,
                source=f"watcher:{collection_name}",
                collection=collection_name,
                _logical_filename=f"{Path(path).name}@{last_offset}",
                metadata={"session_id": session_id, "offset": last_offset},
                probe_config=probe_config,
            )
            self._offsets[path] = actual_end
            logger.info(
                "Append-indexed %s [%d:%d] -> %s (collection: %s)",
                Path(path).name, last_offset, actual_end, doc_id, collection_name,
            )
        except Exception:
            logger.exception("Failed to append-index %s", path)
        finally:
            tmp_path.unlink(missing_ok=True)
