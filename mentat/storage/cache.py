"""Persistent JSON-backed caches for content deduplication and path identity.

ContentHashCache — maps SHA-256 content hashes to doc_ids so identical files
skip the entire indexing pipeline.

PathIndex — maps canonical file paths (or synthetic content keys) to doc_ids
so re-indexing the same resource replaces the old document.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger("mentat.cache")

# 64KB read buffer for hashing large files
_HASH_BUF_SIZE = 65536


def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of file content."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            data = f.read(_HASH_BUF_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


# ── Base class ──────────────────────────────────────────────────────────


class _JsonMap:
    """Persistent str→str map backed by a JSON file.

    Subclasses only need to set ``_filename`` and optionally override
    key-transform methods.
    """

    _filename: str = "map.json"  # override in subclass

    def __init__(self, cache_dir: str = "./mentat_db"):
        self._path = Path(cache_dir) / self._filename
        self._data: Dict[str, str] = self._load()

    # ── persistence ────────────────────────────────────────────────────

    def _load(self) -> Dict[str, str]:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning(f"{self._filename} corrupted, starting fresh.")
                return {}
        return {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    # ── key-level ops ──────────────────────────────────────────────────

    def _get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def _put(self, key: str, value: str):
        self._data[key] = value
        self._save()

    def _remove_key(self, key: str):
        if key in self._data:
            del self._data[key]
            self._save()

    # ── value-level ops ────────────────────────────────────────────────

    def remove(self, doc_id: str):
        """Remove all entries whose value equals *doc_id*."""
        before = len(self._data)
        self._data = {k: v for k, v in self._data.items() if v != doc_id}
        if len(self._data) != before:
            self._save()

    def clear(self):
        self._data = {}
        self._save()

    def __len__(self) -> int:
        return len(self._data)


# ── ContentHashCache ────────────────────────────────────────────────────


class ContentHashCache(_JsonMap):
    """Persistent cache mapping content hashes to doc_ids.

    Checked before probing so duplicate files skip the entire pipeline.
    """

    _filename = "content_hashes.json"

    def get(self, file_path: str) -> Optional[str]:
        """Check if file was already processed. Returns doc_id if cached."""
        content_hash = compute_file_hash(file_path)
        doc_id = self._get(content_hash)
        if doc_id:
            logger.info(f"Cache hit: {file_path} -> {doc_id}")
        return doc_id

    def put(self, file_path: str, doc_id: str):
        """Record that a file with this content was indexed as doc_id."""
        content_hash = compute_file_hash(file_path)
        self._put(content_hash, doc_id)

    def get_by_hash(self, content_hash: str) -> Optional[str]:
        """Look up a doc_id by pre-computed content hash."""
        return self._get(content_hash)

    def put_hash(self, content_hash: str, doc_id: str):
        """Record a mapping from a pre-computed content hash to a doc_id."""
        self._put(content_hash, doc_id)


# ── PathIndex ───────────────────────────────────────────────────────────


class PathIndex(_JsonMap):
    """Persistent index mapping canonical file paths to doc_ids.

    Enables path-based identity: when the same file is re-indexed with
    different content, the old document can be found and replaced instead
    of creating a duplicate.

    For content indexed via ``add_content()`` (no real file path), the
    caller passes a synthetic key like ``__content__:{filename}``.
    """

    _filename = "path_index.json"

    @staticmethod
    def _canon(path: str) -> str:
        """Canonicalize a path key. Synthetic keys (starting with ``__``)
        are kept as-is; real paths are resolved to absolute form."""
        if path.startswith("__"):
            return path
        return str(Path(path).resolve())

    def get(self, path: str) -> Optional[str]:
        """Look up the doc_id previously indexed for *path*."""
        return self._get(self._canon(path))

    def put(self, path: str, doc_id: str):
        """Record that *path* is currently indexed as *doc_id*."""
        self._put(self._canon(path), doc_id)

    def remove_path(self, path: str):
        """Remove entry for a specific path."""
        self._remove_key(self._canon(path))
