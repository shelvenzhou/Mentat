"""Abstract base class for vector storage backends."""

import abc
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mentat.storage.filters import MetadataFilterSet


class BaseVectorStorage(abc.ABC):
    """Interface for vector storage backends.

    Mentat stores two kinds of data:
      - **Stubs**: document-level metadata (one row per doc)
      - **Chunks**: content chunks with embedding vectors (many rows per doc)

    Implementations must provide both stub and chunk operations.
    The chunks table may be created lazily on first ``add_chunks`` call.
    """

    # ── Document-level (stub) operations ─────────────────────────────

    @abc.abstractmethod
    def add_stub(
        self,
        doc_id: str,
        filename: str,
        brief_intro: str,
        instruction: str,
        probe_json: str,
        source: str = "",
        metadata_json: str = "{}",
    ) -> None:
        """Store document-level metadata."""
        ...

    @abc.abstractmethod
    def get_stub(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document stub by ID. Returns None if not found."""
        ...

    @abc.abstractmethod
    def resolve_doc_id(self, prefix: str) -> Optional[str]:
        """Resolve a doc ID prefix to a full ID.

        Returns None if no match.
        Raises ValueError if the prefix is ambiguous (matches multiple docs).
        """
        ...

    @abc.abstractmethod
    def get_doc_ids_by_source(self, source: str) -> List[str]:
        """Return doc IDs matching a source pattern.

        Supports exact match and prefix glob (e.g. "composio:*").
        """
        ...

    @abc.abstractmethod
    def has_chunks(self, doc_id: str) -> bool:
        """Check if a document has chunks stored."""
        ...

    @abc.abstractmethod
    def list_docs(self) -> List[Dict[str, Any]]:
        """List all indexed document stubs."""
        ...

    @abc.abstractmethod
    def count_docs(self) -> int:
        """Count total indexed documents."""
        ...

    @abc.abstractmethod
    def delete_doc(self, doc_id: str) -> None:
        """Delete a document's stub and all its chunks."""
        ...

    # ── Chunk-level operations ───────────────────────────────────────

    @abc.abstractmethod
    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """Store multiple chunks with their vectors.

        On the first call the chunks table may be created with the vector
        dimension inferred from the data.
        """
        ...

    @abc.abstractmethod
    def search(
        self,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 5,
        use_hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
        filters: Optional["MetadataFilterSet"] = None,
    ) -> List[Dict[str, Any]]:
        """Search chunks by vector similarity.

        Args:
            query_vector: Query embedding vector.
            query_text: Raw query text (used for hybrid/FTS search).
            limit: Maximum number of results.
            use_hybrid: Enable hybrid (FTS + vector) search if supported.
            doc_ids: If provided, restrict search to these documents.
            filters: Optional metadata filters to apply as pre-filtering.

        Returns:
            List of chunk dicts with at least: chunk_id, doc_id, filename,
            content, summary, section, chunk_index, _distance.
        """
        ...

    @abc.abstractmethod
    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document, ordered by chunk_index."""
        ...

    @abc.abstractmethod
    def update_chunks(self, doc_id: str, updated_rows: List[Dict[str, Any]]) -> None:
        """Replace all chunks for a document (delete + re-add)."""
        ...

    @abc.abstractmethod
    def count_chunks(self) -> int:
        """Count total stored chunks."""
        ...
