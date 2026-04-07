"""Core orchestrator for the Mentat system.

The Mentat class is the central singleton that wires together all subsystems
and delegates operations to focused modules:
  - Indexer: document ingestion (core/indexer.py)
  - Searcher: query and retrieval (core/searcher.py)
  - Reader: document reading and inspection (core/reader.py)
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from mentat.core.models import (
    MentatResult,
    MentatDocResult,
    ChunkResult,
    MentatConfig,
    Collection,
    BaseAdaptor,
)
from mentat.core.indexer import Indexer
from mentat.core.searcher import Searcher
from mentat.core.reader import Reader
from mentat.core.watcher import MentatWatcher
from mentat.core.embeddings import EmbeddingRegistry
from mentat.core.queue import BackgroundProcessor
from mentat.core.access_tracker import AccessTracker
from mentat.core.section_heat import SectionHeatTracker
from mentat.librarian.engine import Librarian
from mentat.storage.base import BaseVectorStorage
from mentat.storage.vector_db import LanceDBStorage
from mentat.storage.file_store import LocalFileStore
from mentat.storage.cache import ContentHashCache, PathIndex
from mentat.storage.collections import CollectionStore

class Mentat:
    """Core orchestrator for the Mentat system.

    Usage:
        import mentat
        doc_id = await mentat.add("paper.pdf")
        results = await mentat.search("What algorithm?")
        probe_result = mentat.probe("data.csv")
    """

    _instance: Optional["Mentat"] = None

    def __init__(self, config: Optional[MentatConfig] = None):
        from dotenv import load_dotenv
        load_dotenv()

        self.config = config or MentatConfig()
        self.logger = logging.getLogger("mentat")

        # Layer 1: Storage (Haystack)
        self.storage: BaseVectorStorage = LanceDBStorage(db_path=self.config.db_path)
        self.file_store = LocalFileStore(storage_dir=self.config.storage_dir)
        self.cache = ContentHashCache(cache_dir=self.config.db_path)
        self.path_index = PathIndex(cache_dir=self.config.db_path)

        # Layer 3: Librarian
        self.librarian = Librarian(
            summary_model=self.config.summary_model,
            summary_api_key=self.config.summary_api_key or None,
            summary_api_base=self.config.summary_api_base or None,
            enabled=self.config.summary_enabled,
        )

        # Embeddings
        self.embeddings = EmbeddingRegistry.get_provider(
            self.config.embedding_provider,
            model=self.config.embedding_model,
            api_key=self.config.embedding_api_key or None,
            api_base=self.config.embedding_api_base or None,
        )

        # Collections
        self.collections_store = CollectionStore(store_dir=self.config.db_path)

        # Adaptors
        self._adaptors: List[BaseAdaptor] = []

        # Wiki — auto-generated browsable wiki from indexed documents
        from mentat.wiki import WikiGenerator, WikiAdaptor
        self.wiki_generator = WikiGenerator(
            wiki_dir=self.config.wiki_dir,
            storage=self.storage,
            collections_store=self.collections_store,
            file_store=self.file_store,
        )
        self.register_adaptor(WikiAdaptor(self.wiki_generator))

        # Access tracker (two-layer FIFO for on-demand embedding)
        heat_map_path = str(Path(self.config.db_path) / "heat_map.json")
        self.access_tracker = AccessTracker(
            recent_size=self.config.access_recent_size,
            hot_size=self.config.access_hot_size,
            on_promote=self._on_access_promote,
            persist_path=heat_map_path,
        )

        # Section-level heat tracker
        section_heat_path = str(Path(self.config.db_path) / "section_heat_map.json")
        self.section_heat = SectionHeatTracker(
            half_life_seconds=self.config.section_heat_half_life,
            hot_threshold=self.config.section_heat_threshold,
            max_entries=self.config.section_heat_max_entries,
            persist_path=section_heat_path,
        )

        # Background processor (async queue system)
        self.processor = BackgroundProcessor(self, max_concurrent=self.config.max_concurrent_tasks)

        # File watcher (per-collection directory monitoring)
        self.watcher = MentatWatcher(self)

        # Delegate modules
        self.indexer = Indexer(self)
        self.searcher = Searcher(self)
        self.reader = Reader(self)

    # ── Singleton ────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls, config: Optional[MentatConfig] = None) -> "Mentat":
        """Singleton accessor for module-level API."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset singleton (useful for tests)."""
        cls._instance = None

    # ── Lifecycle ────────────────────────────────────────────────────

    def register_adaptor(self, adaptor: BaseAdaptor):
        """Register an adaptor for lifecycle hooks."""
        self._adaptors.append(adaptor)

    async def start(self):
        """Start the background processor and file watcher."""
        await self.processor.start()
        await self.watcher.start()

    async def shutdown(self):
        """Shutdown the file watcher and background processor gracefully."""
        await self.watcher.stop()
        await self.processor.stop()
        self.access_tracker.save_now()
        self.section_heat.save_now()

    # ── Indexing delegates ───────────────────────────────────────────

    async def add(self, *args, **kwargs) -> str:
        return await self.indexer.add(*args, **kwargs)

    async def add_batch(self, *args, **kwargs) -> List[str]:
        return await self.indexer.add_batch(*args, **kwargs)

    async def add_content(self, *args, **kwargs) -> str:
        return await self.indexer.add_content(*args, **kwargs)

    def get_processing_status(self, doc_id: str) -> Dict[str, Any]:
        return self.indexer.get_processing_status(doc_id)

    async def wait_for_completion(self, doc_id: str, timeout: float = 300) -> bool:
        return await self.indexer.wait_for_completion(doc_id, timeout=timeout)

    # ── Search delegates ─────────────────────────────────────────────

    async def search(self, *args, **kwargs) -> List[MentatResult]:
        return await self.searcher.search(*args, **kwargs)

    async def search_grouped(self, *args, **kwargs) -> List[MentatDocResult]:
        return await self.searcher.search_grouped(*args, **kwargs)

    # ── Reader delegates ─────────────────────────────────────────────

    async def get_doc_meta(self, doc_id: str) -> Optional[Dict[str, Any]]:
        return await self.reader.get_doc_meta(doc_id)

    async def inspect(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        return await self.reader.inspect(*args, **kwargs)

    async def summarize_doc(self, doc_id: str) -> bool:
        return await self.reader.summarize_doc(doc_id)

    async def read_structured(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.reader.read_structured(*args, **kwargs)

    async def read_segment(self, *args, **kwargs) -> Dict[str, Any]:
        return await self.reader.read_segment(*args, **kwargs)

    # ── Access tracking ──────────────────────────────────────────────

    async def _on_access_promote(self, key: str) -> None:
        """Callback fired when a key is promoted to the hot queue."""
        path = Path(key)
        if not path.is_file():
            self.logger.debug(f"Hot key is not a file, skipping auto-index: {key}")
            return

        cached_id = self.cache.get(key)
        if cached_id:
            self.logger.info(f"Hot promote: triggering summarization for existing doc {cached_id}")
            await self.summarize_doc(cached_id)
        else:
            self.logger.info(f"Hot promote: indexing new file with summarization: {key}")
            await self.add(key, summarize=True)

    async def track_access(self, path: str) -> Dict[str, Any]:
        """Record a file access event and return tracking status."""
        resolved = str(Path(path).resolve())
        promoted = await self.access_tracker.track(resolved)
        return {
            "path": resolved,
            "promoted": promoted,
            "is_hot": self.access_tracker.is_hot(resolved),
            "is_recent": self.access_tracker.is_recent(resolved),
            "tracker_stats": self.access_tracker.stats(),
        }

    def get_section_heat(
        self,
        doc_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return hottest sections, optionally filtered by document."""
        if doc_id:
            return self.section_heat.get_top_sections(doc_id, limit=limit)
        return self.section_heat.get_hot_sections(limit=limit)

    # ── Collections ──────────────────────────────────────────────────

    def collection(self, name: str) -> "Collection":
        """Get a Collection handle for scoped add/search operations."""
        return Collection(name, self)

    def list_collections(self) -> List[str]:
        """Return all collection names."""
        return self.collections_store.list_collections()

    # ── Stats ────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """Return system statistics."""
        return {
            "docs_indexed": self.storage.count_docs(),
            "chunks_stored": self.storage.count_chunks(),
            "cached_hashes": len(self.cache),
            "storage_size_bytes": self.file_store.total_size(),
            "collections": len(self.collections_store.list_collections()),
            "access_tracker": self.access_tracker.stats(),
            "section_heat": self.section_heat.stats(),
        }
