import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING

from mentat.storage.base import BaseVectorStorage

if TYPE_CHECKING:
    from mentat.storage.filters import MetadataFilterSet


# Schema for document-level metadata (stubs)
STUBS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("brief_intro", pa.string()),
        pa.field("instruction", pa.string()),
        pa.field("probe_json", pa.string()),  # Full ProbeResult as JSON
        pa.field("source", pa.string()),  # Origin tag, e.g. "web_fetch", "composio:gmail"
        pa.field("metadata_json", pa.string()),  # Arbitrary JSON metadata from caller
    ]
)


class LanceDBStorage(BaseVectorStorage):
    """Storage layer using LanceDB for vector search and FTS.

    Stores document stubs (metadata) and chunks (content + vectors) in
    separate tables.  The chunks table is created **lazily** on the first
    ``add_chunks`` call so the vector dimension is inferred from the actual
    embedding output — no upfront configuration needed.
    """

    def __init__(self, db_path: str = "./mentat_db"):
        self.db_path = db_path
        self.db = lancedb.connect(db_path)
        self.stubs_table = self._get_or_create_table("stubs", STUBS_SCHEMA)
        # Opened lazily — see _ensure_chunks_table()
        self._chunks_table = None
        self._indexes_ensured = False

    def _table_names(self) -> Set[str]:
        """Return existing table names as a set of strings.

        Handles both new (``list_tables`` → ``ListTablesResponse``) and old
        (``table_names`` → ``List[str]``) lancedb APIs.
        """
        try:
            raw = self.db.list_tables()
            # New API: ListTablesResponse with a .tables attribute
            if hasattr(raw, "tables"):
                return set(raw.tables)
            # Fallback: older versions may return a plain list
            return {item if isinstance(item, str) else getattr(item, "name", str(item)) for item in raw}
        except AttributeError:
            return set(self.db.table_names())  # type: ignore[attr-defined]

    @property
    def chunks_table(self):
        """Return the chunks table, opening an existing one if needed."""
        if self._chunks_table is None:
            if "chunks" in self._table_names():
                self._chunks_table = self.db.open_table("chunks")
                self._ensure_indexes()
        return self._chunks_table

    def _ensure_chunks_table(self, vector_dim: int):
        """Create the chunks table if it doesn't exist yet."""
        if self._chunks_table is not None:
            return
        if "chunks" in self._table_names():
            self._chunks_table = self.db.open_table("chunks")
            return
        schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("content", pa.string()),
                pa.field("summary", pa.string()),
                pa.field("section", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), vector_dim)),
                # Fields for split chunks (optional, null if chunk wasn't split)
                pa.field("is_split", pa.bool_(), nullable=True),
                pa.field("piece_index", pa.int32(), nullable=True),
                pa.field("total_pieces", pa.int32(), nullable=True),
                # Metadata fields for pre-filtering (duplicated from stub for speed)
                pa.field("source", pa.string(), nullable=True),
                pa.field("indexed_at", pa.float64(), nullable=True),
                pa.field("file_type", pa.string(), nullable=True),
                pa.field("metadata_json", pa.string(), nullable=True),
            ]
        )
        self._chunks_table = self.db.create_table("chunks", schema=schema)

    def _get_or_create_table(self, name: str, schema: pa.Schema):
        if name in self._table_names():
            return self.db.open_table(name)
        return self.db.create_table(name, schema=schema)

    # --- Document-level operations ---

    def add_stub(
        self,
        doc_id: str,
        filename: str,
        brief_intro: str,
        instruction: str,
        probe_json: str,
        source: str = "",
        metadata_json: str = "{}",
    ):
        """Store document-level metadata."""
        data = [
            {
                "id": doc_id,
                "filename": filename,
                "brief_intro": brief_intro,
                "instruction": instruction,
                "probe_json": probe_json,
                "source": source,
                "metadata_json": metadata_json,
            }
        ]
        self.stubs_table.add(data)

    def get_stub(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document stub by ID."""
        res = self.stubs_table.search().where(f"id = '{doc_id}'").limit(1).to_list()
        if res:
            return res[0]
        return None

    def resolve_doc_id(self, prefix: str) -> Optional[str]:
        """Resolve a doc ID prefix to a full ID. Returns None if no match, raises ValueError if ambiguous."""
        docs = self.list_docs()
        matches = [d["id"] for d in docs if d.get("id", "").startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous prefix '{prefix}', matches {len(matches)} documents: "
                + ", ".join(m[:12] + "…" for m in matches[:5])
            )
        return None

    def get_doc_ids_by_source(self, source: str) -> List[str]:
        """Return doc IDs matching a source pattern.

        Supports exact match and prefix glob (e.g. "composio:*" matches
        "composio:gmail", "composio:notion", etc.).
        """
        try:
            if source.endswith("*"):
                prefix = source[:-1]
                rows = (
                    self.stubs_table.search()
                    .where(f"source >= '{prefix}' AND source < '{prefix}~'")
                    .limit(10000)
                    .to_list()
                )
            else:
                rows = (
                    self.stubs_table.search()
                    .where(f"source = '{source}'")
                    .limit(10000)
                    .to_list()
                )
            return [r["id"] for r in rows]
        except Exception:
            return []

    def has_chunks(self, doc_id: str) -> bool:
        """Check if a document has chunks stored.

        Useful for determining if background processing has completed.

        Args:
            doc_id: Document identifier

        Returns:
            True if the document has at least one chunk with vectors
        """
        if self.chunks_table is None:
            return False

        try:
            res = (
                self.chunks_table.search()
                .where(f"doc_id = '{doc_id}'")
                .limit(1)
                .to_list()
            )
            return len(res) > 0
        except Exception:
            return False

    # --- Chunk-level operations ---

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Store multiple chunks with their vectors.

        On the first call the chunks table is created with the vector
        dimension inferred from the data.  Indexes (FTS + scalar) are
        ensured once per session after the first batch is written.
        """
        if not chunks:
            return
        vector_dim = len(chunks[0]["vector"])
        self._ensure_chunks_table(vector_dim)
        self.chunks_table.add(chunks)
        self._ensure_indexes()

    def search(
        self,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 5,
        use_hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
        filters: Optional["MetadataFilterSet"] = None,
    ) -> List[Dict[str, Any]]:
        """Search chunks by vector similarity, optionally with hybrid (FTS + vector) search.

        Args:
            doc_ids: If provided, restrict search to chunks belonging to these documents.
                     Uses LanceDB pre-filtering for efficient scoped search.
            filters: Optional MetadataFilterSet for metadata pre-filtering.
        """
        if self.chunks_table is None:
            return []

        if use_hybrid and self._has_fts_index() and query_text:
            # LanceDB hybrid: search(None) + explicit vector() and text()
            q = (
                self.chunks_table.search(query_type="hybrid")
                .vector(query_vector)
                .text(query_text)
            )
        else:
            # Pure vector search
            q = self.chunks_table.search(query_vector)

        # Build WHERE clause from doc_ids and metadata filters
        where_parts: List[str] = []
        if doc_ids is not None:
            ids_str = ", ".join(f"'{d}'" for d in doc_ids)
            where_parts.append(f"doc_id IN ({ids_str})")
        if filters is not None and not filters.is_empty():
            where_parts.append(filters.to_lance_sql())

        if where_parts:
            q = q.where(" AND ".join(where_parts))

        return q.limit(limit).to_list()

    def _has_fts_index(self) -> bool:
        """Check if FTS index exists on chunks table."""
        try:
            indices = self.chunks_table.list_indices()
            return any(getattr(idx, "index_type", "") == "FTS" for idx in indices)
        except Exception:
            return False

    def _ensure_indexes(self):
        """Create FTS and scalar indexes if not already ensured this session."""
        if self._indexes_ensured:
            return
        if self._chunks_table is None:
            return
        self.create_fts_index()
        self.ensure_scalar_index()
        self._indexes_ensured = True

    def create_fts_index(self):
        """Create FTS index on chunk content for hybrid search."""
        try:
            self.chunks_table.create_fts_index("content", replace=True)
        except Exception:
            pass  # Index may already exist

    def ensure_scalar_index(self):
        """Create scalar index on doc_id for fast collection pre-filtering."""
        try:
            self.chunks_table.create_index("doc_id", index_type="BTREE", replace=True)
        except Exception:
            pass  # Index may already exist

    def get_chunks_by_doc(self, doc_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document, ordered by chunk_index."""
        try:
            rows = (
                self.chunks_table.search()
                .where(f"doc_id = '{doc_id}'")
                .limit(10000)
                .to_list()
            )
            rows.sort(key=lambda r: r.get("chunk_index", 0))
            return rows
        except Exception:
            return []

    def update_chunks(self, doc_id: str, updated_rows: List[Dict[str, Any]]) -> None:
        """Replace all chunks for a document (delete + re-add).

        LanceDB has no row-level UPDATE, so we delete existing chunks
        for the doc_id and re-add the updated rows.
        """
        if self.chunks_table is None or not updated_rows:
            return

        # Delete existing chunks for this document
        self.chunks_table.delete(f"doc_id = '{doc_id}'")

        # Strip LanceDB internal fields before re-adding
        clean = []
        for row in updated_rows:
            clean.append({k: v for k, v in row.items() if not k.startswith("_")})

        self.chunks_table.add(clean)

    def count_docs(self) -> int:
        """Count total indexed documents."""
        try:
            return self.stubs_table.count_rows()
        except Exception:
            return 0

    def count_chunks(self) -> int:
        """Count total stored chunks."""
        try:
            return self.chunks_table.count_rows()
        except Exception:
            return 0

    def delete_doc(self, doc_id: str):
        """Delete a document's stub and all its chunks."""
        try:
            self.stubs_table.delete(f"id = '{doc_id}'")
        except Exception:
            pass
        if self.chunks_table is not None:
            try:
                self.chunks_table.delete(f"doc_id = '{doc_id}'")
            except Exception:
                pass

    def list_docs(self) -> List[Dict[str, Any]]:
        """List all indexed document stubs."""
        try:
            return self.stubs_table.search().limit(1000).to_list()
        except Exception:
            return []
