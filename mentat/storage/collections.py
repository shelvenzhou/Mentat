"""Persistent collection store — named sets of doc_id references.

Collections are lightweight groupings over the shared document store.
A document can belong to multiple collections without data duplication.
Stored as a simple JSON file alongside the database.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger("mentat.collections")


class CollectionStore:
    """Manages named collections of doc_id references.

    Persistence format (collections.json):
        {"collection_name": ["doc_id_1", "doc_id_2", ...], ...}
    """

    def __init__(self, store_dir: str = "./mentat_db"):
        self._path = Path(store_dir) / "collections.json"
        self._data: Dict[str, List[str]] = self._load()

    def _load(self) -> Dict[str, List[str]]:
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("Collections file corrupted, starting fresh.")
                return {}
        return {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def list_collections(self) -> List[str]:
        """Return all collection names."""
        return list(self._data.keys())

    def get_doc_ids(self, name: str) -> Set[str]:
        """Return doc_ids in a collection (empty set if not found)."""
        return set(self._data.get(name, []))

    def add_doc(self, name: str, doc_id: str):
        """Add a doc_id to a collection (creates collection if needed)."""
        if name not in self._data:
            self._data[name] = []
        if doc_id not in self._data[name]:
            self._data[name].append(doc_id)
            self._save()

    def remove_doc(self, name: str, doc_id: str):
        """Remove a doc_id from a collection."""
        if name in self._data and doc_id in self._data[name]:
            self._data[name].remove(doc_id)
            self._save()

    def delete_collection(self, name: str) -> bool:
        """Delete an entire collection (not the underlying documents)."""
        if name in self._data:
            del self._data[name]
            self._save()
            return True
        return False

    def doc_collections(self, doc_id: str) -> List[str]:
        """Return all collections a doc_id belongs to."""
        return [name for name, ids in self._data.items() if doc_id in ids]
