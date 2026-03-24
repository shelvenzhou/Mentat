"""Data models, configuration, and base interfaces for the Mentat system."""

import abc
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mentat.core.hub import Mentat

from mentat.probes.base import TocEntry


class MentatResult(BaseModel):
    """A search result returned by Mentat."""

    doc_id: str
    chunk_id: str
    filename: str
    section: Optional[str] = None
    content: str = ""
    summary: str = ""
    brief_intro: str = ""
    instructions: str = ""
    score: float = 0.0
    toc_entries: List[TocEntry] = Field(default_factory=list)
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkResult(BaseModel):
    """A single chunk within a grouped search result."""

    chunk_id: str
    section: Optional[str] = None
    content: str = ""
    summary: str = ""
    score: float = 0.0


class MentatDocResult(BaseModel):
    """Search results grouped by document — doc-level metadata once,
    all matching chunks nested."""

    doc_id: str
    filename: str
    brief_intro: str = ""
    instructions: str = ""
    source: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    toc_entries: List[TocEntry] = Field(default_factory=list)
    score: float = 0.0
    chunks: List[ChunkResult] = Field(default_factory=list)


@dataclass
class MentatConfig:
    """Configuration for Mentat instance.

    Values are resolved in order: explicit argument > environment variable > default.
    Environment variables use the ``MENTAT_`` prefix (see .env.example).
    """

    # Storage
    db_path: str = ""
    storage_dir: str = ""

    # Summary model — Phase 1: bulk chunk summarisation (use a fast/cheap model)
    summary_model: str = ""
    summary_api_key: str = ""
    summary_api_base: str = ""

    # Embedding model
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_api_base: str = ""

    # Background processing
    max_concurrent_tasks: int = 5

    # Access tracking
    access_recent_size: int = 200
    access_hot_size: int = 50

    # Chunk normalization
    chunk_target_tokens: int = 1000

    # Section heat tracking
    section_heat_half_life: float = 86400.0
    section_heat_threshold: float = 5.0
    section_heat_max_entries: int = 1000

    def __post_init__(self):
        self.db_path = self.db_path or os.getenv("MENTAT_DB_PATH", "./mentat_db")
        self.storage_dir = self.storage_dir or os.getenv(
            "MENTAT_STORAGE_DIR", "./mentat_files"
        )

        # Summary model (fast/cheap — bulk chunk summarisation)
        self.summary_model = self.summary_model or os.getenv(
            "MENTAT_SUMMARY_MODEL", "gpt-4o-mini"
        )
        self.summary_api_key = self.summary_api_key or os.getenv(
            "MENTAT_SUMMARY_API_KEY", ""
        )
        self.summary_api_base = self.summary_api_base or os.getenv(
            "MENTAT_SUMMARY_API_BASE", ""
        )

        # Embedding
        self.embedding_provider = self.embedding_provider or os.getenv(
            "MENTAT_EMBEDDING_PROVIDER", "litellm"
        )
        self.embedding_model = self.embedding_model or os.getenv(
            "MENTAT_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.embedding_api_key = self.embedding_api_key or os.getenv(
            "MENTAT_EMBEDDING_API_KEY", ""
        )
        self.embedding_api_base = self.embedding_api_base or os.getenv(
            "MENTAT_EMBEDDING_API_BASE", ""
        )

        # Background processing
        self.max_concurrent_tasks = int(
            os.getenv("MENTAT_MAX_CONCURRENT_TASKS", str(self.max_concurrent_tasks))
        )

        # Access tracking
        self.access_recent_size = int(
            os.getenv("MENTAT_ACCESS_RECENT_SIZE", str(self.access_recent_size))
        )
        self.access_hot_size = int(
            os.getenv("MENTAT_ACCESS_HOT_SIZE", str(self.access_hot_size))
        )

        # Chunk normalization
        self.chunk_target_tokens = int(
            os.getenv("MENTAT_CHUNK_TARGET_TOKENS", str(self.chunk_target_tokens))
        )

        # Section heat tracking
        self.section_heat_half_life = float(
            os.getenv("MENTAT_SECTION_HEAT_HALF_LIFE", str(self.section_heat_half_life))
        )
        self.section_heat_threshold = float(
            os.getenv("MENTAT_SECTION_HEAT_THRESHOLD", str(self.section_heat_threshold))
        )
        self.section_heat_max_entries = int(
            os.getenv("MENTAT_SECTION_HEAT_MAX_ENTRIES", str(self.section_heat_max_entries))
        )


class BaseAdaptor(abc.ABC):
    """Base class for integrating Mentat with external systems.

    An adaptor can hook into Mentat's lifecycle:
    - on_document_indexed: called after a document is indexed
    - on_search_results: called after search, can transform/filter results
    - transform_query: called before search, can rewrite queries
    """

    @abc.abstractmethod
    def on_document_indexed(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Called after a document is successfully indexed."""
        ...

    @abc.abstractmethod
    def on_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Called after search. Can transform or filter results."""
        ...

    def transform_query(self, query: str) -> str:
        """Optional: transform query before search. Default is pass-through."""
        return query


class Collection:
    """A named group of documents — scoped view over shared storage.

    Acts as a thin wrapper around Mentat. Documents are indexed into the
    shared store; the collection just holds doc_id references.
    """

    def __init__(self, name: str, mentat: "Mentat"):
        self.name = name
        self._mentat = mentat
        self._store = mentat.collections_store

    @property
    def doc_ids(self) -> set:
        return self._store.get_doc_ids(self.name)

    async def add(
        self,
        path: str,
        force: bool = False,
        summarize: bool = False,
        use_llm_instructions: bool = False,
        wait: bool = False,
        source: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Index a file (if needed) and add it to this collection.

        Uses the shared cache — if the file was already indexed,
        just links the existing doc_id without re-processing.
        """
        doc_id = await self._mentat.add(
            path,
            force=force,
            summarize=summarize,
            use_llm_instructions=use_llm_instructions,
            wait=wait,
            source=source,
            metadata=metadata,
            collection=self.name,
        )
        return doc_id

    async def search(
        self, query: str, top_k: int = 5, hybrid: bool = False
    ) -> List[MentatResult]:
        """Search only within this collection's documents."""
        ids = list(self.doc_ids)
        if not ids:
            return []
        return await self._mentat.search(
            query, top_k=top_k, hybrid=hybrid, doc_ids=ids
        )

    def remove(self, doc_id: str):
        """Remove a document from this collection (does NOT delete from storage)."""
        self._store.remove_doc(self.name, doc_id)

    def list_docs(self) -> List[Dict[str, Any]]:
        """List documents in this collection with their metadata."""
        docs = []
        for doc_id in self.doc_ids:
            stub = self._mentat.storage.get_stub(doc_id)
            if stub:
                docs.append(
                    {
                        "doc_id": doc_id,
                        "filename": stub.get("filename", ""),
                        "brief_intro": stub.get("brief_intro", ""),
                    }
                )
        return docs

    def delete(self) -> bool:
        """Delete this collection (does NOT delete underlying documents)."""
        return self._store.delete_collection(self.name)
