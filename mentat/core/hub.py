import logging
import uuid
import json
import asyncio
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

from mentat.core.telemetry import Telemetry
from mentat.core.embeddings import EmbeddingRegistry
from mentat.probes.csv_probe import CSVProbe
from mentat.probes.markdown_probe import MarkdownProbe
from mentat.probes.pdf_probe import PDFProbe
from mentat.probes.json_probe import JSONProbe
from mentat.probes.web_probe import WebProbe
from mentat.probes.code_probe import CodeProbe
from mentat.librarian.engine import Librarian
from mentat.storage.vector_db import LanceDBStorage
from mentat.storage.file_store import FileStore


class MentatResult(BaseModel):
    id: str
    filename: str
    brief_intro: str
    instructions: str
    score: float = 0.0


class Mentat:
    def __init__(
        self, db_path: str = "./mentat_db", storage_dir: str = "./mentat_files"
    ):
        self.logger = logging.getLogger("mentat")
        self.probes = [
            CSVProbe(),
            MarkdownProbe(),
            PDFProbe(),
            JSONProbe(),
            WebProbe(),
            CodeProbe(),
        ]
        self.librarian = Librarian()
        self.storage = LanceDBStorage(db_path=db_path)
        self.file_store = FileStore(storage_dir=storage_dir)
        self.embeddings = EmbeddingRegistry.get_provider("litellm")

    async def add(self, path: str, metadata: dict = None) -> str:
        doc_id = str(uuid.uuid4())
        filename = Path(path).name
        self.logger.info(f"Adding file: {path} (ID: {doc_id})")

        # 1. Probe
        probe_result = None
        with Telemetry.time_it(doc_id, "probe"):
            # Simple content type detection by extension for now
            ext = Path(path).suffix.lower()
            for p in self.probes:
                if p.can_handle(filename, ""):  # Simplified check
                    probe_result = p.run(path)
                    break

        if not probe_result:
            raise ValueError(f"No probe found for file: {path}")

        probe_result.doc_id = doc_id

        # 2. Librarian
        with Telemetry.time_it(doc_id, "librarian"):
            brief_intro, instructions, tokens = await self.librarian.generate_guide(
                probe_result
            )
            Telemetry.record_tokens(doc_id, tokens)

        # 3. Embedding
        # Embed the summary + hint as the "stub" representative
        stub_text = f"{brief_intro}\n{probe_result.summary_hint}"
        vector = await self.embeddings.embed(stub_text)

        # 4. Store
        self.storage.add_stub(
            doc_id=doc_id,
            filename=filename,
            content=stub_text,
            vector=vector,
            metadata=json.dumps(probe_result.model_dump()),
            instruction=instructions,
        )

        # Raw storage
        self.file_store.save_file(path, doc_id)

        # Telemetry savings calculation
        # Simplified: compare stub text size to original file size (approx)
        original_size = os.path.getsize(path)
        stub_size = len(stub_text.encode("utf-8"))
        savings = 1.0 - (stub_size / max(1, original_size))
        Telemetry.record_savings(doc_id, savings)

        print(Telemetry.format_stats(doc_id))
        return doc_id

    async def search(self, query: str, strategy: str = "auto") -> List[MentatResult]:
        query_vector = await self.embeddings.embed(query)
        raw_results = self.storage.search_hybrid(query_vector, query)

        results = []
        for r in raw_results:
            results.append(
                MentatResult(
                    id=r["id"],
                    filename=r["filename"],
                    brief_intro=r["content"].split("\n")[
                        0
                    ],  # First line is intro usually
                    instructions=r["instruction"],
                    score=r.get("_distance", 0.0),  # Note: LanceDB distance
                )
            )
        return results

    async def inspect(self, doc_id: str) -> dict:
        # Search by ID in LanceDB
        tbl = self.storage.table
        res = tbl.search().where(f"id = '{doc_id}'").to_list()
        if not res:
            return {}
        return res[0]
