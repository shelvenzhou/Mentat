import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional, Set


# Schema for document-level metadata (stubs)
STUBS_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string()),
        pa.field("filename", pa.string()),
        pa.field("brief_intro", pa.string()),
        pa.field("instruction", pa.string()),
        pa.field("probe_json", pa.string()),  # Full ProbeResult as JSON
    ]
)


class LanceDBStorage:
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

    def _table_names(self) -> Set[str]:
        """Return existing table names as a set of strings.

        Handles both old (``table_names`` → List[str]) and new
        (``list_tables`` → List[TableInfo]) lancedb APIs.
        """
        try:
            raw = self.db.table_names()
        except AttributeError:
            raw = self.db.list_tables()
        # Normalise: entries may be strings or objects with a .name attr
        names: Set[str] = set()
        for item in raw:
            names.add(item if isinstance(item, str) else str(item))
        return names

    @property
    def chunks_table(self):
        """Return the chunks table, opening an existing one if needed."""
        if self._chunks_table is None:
            if "chunks" in self._table_names():
                self._chunks_table = self.db.open_table("chunks")
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
    ):
        """Store document-level metadata."""
        data = [
            {
                "id": doc_id,
                "filename": filename,
                "brief_intro": brief_intro,
                "instruction": instruction,
                "probe_json": probe_json,
            }
        ]
        self.stubs_table.add(data)

    def get_stub(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document stub by ID."""
        res = self.stubs_table.search().where(f"id = '{doc_id}'").limit(1).to_list()
        if res:
            return res[0]
        return None

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
        dimension inferred from the data.
        """
        if not chunks:
            return
        vector_dim = len(chunks[0]["vector"])
        self._ensure_chunks_table(vector_dim)
        self.chunks_table.add(chunks)

    def search(
        self,
        query_vector: List[float],
        query_text: str = "",
        limit: int = 5,
        use_hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search chunks by vector similarity, optionally with hybrid (FTS + vector) search.

        Args:
            doc_ids: If provided, restrict search to chunks belonging to these documents.
                     Uses LanceDB pre-filtering for efficient scoped search.
        """
        if use_hybrid and self._has_fts_index():
            # LanceDB hybrid search: vector + FTS with reranking
            q = self.chunks_table.search(query_vector, query_type="hybrid")
        else:
            # Pure vector search
            q = self.chunks_table.search(query_vector)

        if doc_ids is not None:
            ids_str = ", ".join(f"'{d}'" for d in doc_ids)
            q = q.where(f"doc_id IN ({ids_str})")

        return q.limit(limit).to_list()

    def _has_fts_index(self) -> bool:
        """Check if FTS index exists on chunks table."""
        try:
            indices = self.chunks_table.list_indices()
            return any(getattr(idx, "index_type", "") == "FTS" for idx in indices)
        except Exception:
            return False

    def create_fts_index(self):
        """Create FTS index on chunk content for hybrid search."""
        try:
            self.chunks_table.create_fts_index("content")
        except Exception:
            pass  # Index may already exist

    def ensure_scalar_index(self):
        """Create scalar index on doc_id for fast collection pre-filtering."""
        try:
            self.chunks_table.create_index("doc_id", index_type="BTREE")
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

    def list_docs(self) -> List[Dict[str, Any]]:
        """List all indexed document stubs."""
        try:
            return self.stubs_table.search().limit(1000).to_list()
        except Exception:
            return []
