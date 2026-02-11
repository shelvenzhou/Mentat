import lancedb
import pyarrow as pa
from typing import List, Dict, Any, Optional
import os


class LanceDBStorage:
    def __init__(self, db_path: str = "./mentat_db", table_name: str = "stubs"):
        self.db_path = db_path
        self.table_name = table_name
        self.db = lancedb.connect(db_path)
        self.table = self._get_or_create_table()

    def _get_or_create_table(self):
        # Define schema for the "stubs"
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("content", pa.string()),  # The stub/chunk content
                pa.field(
                    "vector", pa.list_(pa.float32(), 1536)
                ),  # Assuming 1536 for OpenAI-like, will be dynamic later
                pa.field("metadata", pa.string()),  # JSON string
                pa.field("instruction", pa.string()),
            ]
        )

        if self.table_name in self.db.table_names():
            return self.db.open_table(self.table_name)
        else:
            # Create table with dummy record to enforce schema if needed, or just let it be dynamic
            return self.db.create_table(self.table_name, schema=schema)

    def add_stub(
        self,
        doc_id: str,
        filename: str,
        content: str,
        vector: List[float],
        metadata: str,
        instruction: str,
    ):
        data = [
            {
                "id": doc_id,
                "filename": filename,
                "content": content,
                "vector": vector,
                "metadata": metadata,
                "instruction": instruction,
            }
        ]
        self.table.add(data)

    def search_hybrid(
        self, query_vector: List[float], query_text: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        # LanceDB hybrid search (Vector + FTS)
        # Note: FTS needs to be enabled on the table first
        results = self.table.search(query_vector).limit(limit).to_list()
        return results

    def create_fts_index(self):
        self.table.create_fts_index("content")
