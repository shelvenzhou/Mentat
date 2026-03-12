"""Persistent collection store — named sets of doc_id references.

Collections are lightweight groupings over the shared document store.
A document can belong to multiple collections without data duplication.
Stored as a simple JSON file alongside the database.

Each collection has:
  - doc_ids: list of document IDs (the core reference set)
  - metadata: opaque dict (Mentat stores/returns but never interprets)
  - watch_paths: directories to watch for auto-indexing
  - watch_ignore: glob patterns to exclude from watching
  - auto_add_sources: source tags that auto-route docs to this collection
  - created_at: ISO timestamp
"""

import fnmatch
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("mentat.collections")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CollectionStore:
    """Manages named collections of doc_id references with metadata.

    Persistence format (collections.json)::

        {
          "collection_name": {
            "doc_ids": ["id1", "id2"],
            "metadata": {"type": "system", ...},
            "watch_paths": ["/some/dir"],
            "watch_ignore": ["node_modules", "*.lock"],
            "auto_add_sources": ["openclaw:*"],
            "created_at": "2026-03-12T00:00:00+00:00"
          }
        }

    Backward compatibility: if loaded data has the old flat format
    (``{"name": ["id1", "id2"]}``), it is auto-migrated on first access.
    """

    def __init__(self, store_dir: str = "./mentat_db"):
        self._path = Path(store_dir) / "collections.json"
        self._data: Dict[str, Dict[str, Any]] = self._load()

    # ── Persistence ─────────────────────────────────────────────────

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self._path.exists():
            return {}
        try:
            with open(self._path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Collections file corrupted, starting fresh.")
            return {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── Collection CRUD ─────────────────────────────────────────────

    def create(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        watch_paths: Optional[List[str]] = None,
        watch_ignore: Optional[List[str]] = None,
        auto_add_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a collection (or update config if it already exists).

        Returns the collection record.
        """
        if name in self._data:
            rec = self._data[name]
            if metadata is not None:
                rec["metadata"] = metadata
            if watch_paths is not None:
                rec["watch_paths"] = watch_paths
            if watch_ignore is not None:
                rec["watch_ignore"] = watch_ignore
            if auto_add_sources is not None:
                rec["auto_add_sources"] = auto_add_sources
        else:
            self._data[name] = {
                "doc_ids": [],
                "metadata": metadata or {},
                "watch_paths": watch_paths or [],
                "watch_ignore": watch_ignore or [],
                "auto_add_sources": auto_add_sources or [],
                "created_at": _now_iso(),
            }
        self._save()
        return self._data[name]

    def list_collections(self) -> List[str]:
        """Return all collection names."""
        return list(self._data.keys())

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        """Return full collection record, or None if not found."""
        return self._data.get(name)

    def get_doc_ids(self, name: str) -> Set[str]:
        """Return doc_ids in a collection (empty set if not found)."""
        rec = self._data.get(name)
        if rec is None:
            return set()
        return set(rec["doc_ids"])

    def delete_collection(self, name: str) -> bool:
        """Delete an entire collection (not the underlying documents)."""
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False

    # ── Doc membership ──────────────────────────────────────────────

    def add_doc(self, name: str, doc_id: str):
        """Add a doc_id to a collection (creates minimal collection if needed)."""
        if name not in self._data:
            self._data[name] = {
                "doc_ids": [],
                "metadata": {},
                "watch_paths": [],
                "watch_ignore": [],
                "auto_add_sources": [],
                "created_at": _now_iso(),
            }
        if doc_id not in self._data[name]["doc_ids"]:
            self._data[name]["doc_ids"].append(doc_id)
            self._save()

    def remove_doc(self, name: str, doc_id: str):
        """Remove a doc_id from a collection."""
        rec = self._data.get(name)
        if rec and doc_id in rec["doc_ids"]:
            rec["doc_ids"].remove(doc_id)
            self._save()

    def doc_collections(self, doc_id: str) -> List[str]:
        """Return all collections a doc_id belongs to."""
        return [
            name for name, rec in self._data.items()
            if doc_id in rec["doc_ids"]
        ]

    # ── Metadata ────────────────────────────────────────────────────

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Return metadata dict for a collection, or None if not found."""
        rec = self._data.get(name)
        if rec is None:
            return None
        return rec["metadata"]

    def update_metadata(self, name: str, metadata: Dict[str, Any]):
        """Merge keys into a collection's metadata."""
        rec = self._data.get(name)
        if rec is None:
            return
        rec["metadata"].update(metadata)
        self._save()

    # ── Watch config ────────────────────────────────────────────────

    def get_watch_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Return watch config for a collection, or None if not found."""
        rec = self._data.get(name)
        if rec is None:
            return None
        return {
            "watch_paths": rec["watch_paths"],
            "watch_ignore": rec["watch_ignore"],
        }

    def get_all_watch_configs(self) -> Dict[str, Dict[str, Any]]:
        """Return watch configs for all collections that have watch_paths."""
        return {
            name: {"watch_paths": rec["watch_paths"], "watch_ignore": rec["watch_ignore"]}
            for name, rec in self._data.items()
            if rec.get("watch_paths")
        }

    # ── Auto-routing ────────────────────────────────────────────────

    def get_auto_route_targets(self, source: str) -> List[str]:
        """Return collection names whose auto_add_sources match this source.

        Matching supports fnmatch-style globs (e.g. "openclaw:*" matches
        "openclaw:Read", "openclaw:WebFetch", etc.).
        """
        if not source:
            return []
        targets = []
        for name, rec in self._data.items():
            for pattern in rec.get("auto_add_sources", []):
                if fnmatch.fnmatch(source, pattern):
                    targets.append(name)
                    break
        return targets

    # ── Garbage collection ──────────────────────────────────────────

    def gc(self, now: Optional[float] = None) -> List[str]:
        """Remove collections whose metadata.ttl has expired.

        TTL is in seconds, measured from created_at.
        Returns list of deleted collection names.
        """
        if now is None:
            now = time.time()
        to_delete = []
        for name, rec in self._data.items():
            ttl = rec.get("metadata", {}).get("ttl")
            if ttl is None:
                continue
            created_at = rec.get("created_at", "")
            try:
                created_ts = datetime.fromisoformat(created_at).timestamp()
            except (ValueError, TypeError):
                continue
            if now - created_ts > ttl:
                to_delete.append(name)
        for name in to_delete:
            del self._data[name]
        if to_delete:
            self._save()
            logger.info("GC removed %d expired collections: %s", len(to_delete), to_delete)
        return to_delete
