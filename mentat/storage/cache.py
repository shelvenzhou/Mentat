"""Global content-hash cache to avoid reprocessing identical files.

Computes SHA-256 of file content and maps it to existing doc_ids.
Checked before probing so duplicate files skip the entire pipeline.
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


class ContentHashCache:
    """Persistent cache mapping content hashes to doc_ids.

    Stored as a simple JSON file alongside the database.
    """

    def __init__(self, cache_dir: str = "./mentat_db"):
        self._cache_path = Path(cache_dir) / "content_hashes.json"
        self._cache: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if self._cache_path.exists():
            try:
                with open(self._cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Cache file corrupted, starting fresh.")
                return {}
        return {}

    def _save(self):
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._cache_path, "w") as f:
            json.dump(self._cache, f, indent=2)

    def get(self, file_path: str) -> Optional[str]:
        """Check if file was already processed. Returns doc_id if cached, None otherwise."""
        content_hash = compute_file_hash(file_path)
        doc_id = self._cache.get(content_hash)
        if doc_id:
            logger.info(f"Cache hit: {file_path} -> {doc_id}")
        return doc_id

    def put(self, file_path: str, doc_id: str):
        """Record that a file with this content was indexed as doc_id."""
        content_hash = compute_file_hash(file_path)
        self._cache[content_hash] = doc_id
        self._save()

    def get_by_hash(self, content_hash: str) -> Optional[str]:
        """Look up a doc_id by pre-computed content hash."""
        return self._cache.get(content_hash)

    def put_hash(self, content_hash: str, doc_id: str):
        """Record a mapping from a pre-computed content hash to a doc_id."""
        self._cache[content_hash] = doc_id
        self._save()

    def remove(self, doc_id: str):
        """Remove a doc_id from cache (e.g. on re-index)."""
        self._cache = {h: d for h, d in self._cache.items() if d != doc_id}
        self._save()

    def clear(self):
        """Clear the entire cache."""
        self._cache = {}
        self._save()

    def __len__(self) -> int:
        return len(self._cache)
