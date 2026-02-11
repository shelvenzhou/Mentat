import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional


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

# Schema for chunk-level data with vectors
CHUNKS_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("doc_id", pa.string()),  # Foreign key to stubs
        pa.field("filename", pa.string()),
        pa.field("content", pa.string()),
        pa.field("section", pa.string()),
        pa.field("chunk_index", pa.int32()),
        pa.field(
            "vector", pa.list_(pa.float32(), 1536)
        ),  # Dimension configurable later
    ]
)


class LanceDBStorage:
    """Storage layer using LanceDB for vector search and FTS.
    Stores document stubs (metadata) and chunks (content + vectors) in separate tables.
    """

    def __init__(self, db_path: str = "./mentat_db", vector_dim: int = 1536):
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.db = lancedb.connect(db_path)
        self.stubs_table = self._get_or_create_table("stubs", STUBS_SCHEMA)
        self.chunks_table = self._get_or_create_table(
            "chunks", self._chunks_schema(vector_dim)
        )

    def _chunks_schema(self, dim: int) -> pa.Schema:
        return pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("content", pa.string()),
                pa.field("section", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

    def _get_or_create_table(self, name: str, schema: pa.Schema):
        if name in self.db.table_names():
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

    # --- Chunk-level operations ---

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Store multiple chunks with their vectors."""
        if chunks:
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
