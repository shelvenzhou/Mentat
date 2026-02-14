import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel

from dotenv import load_dotenv

from mentat.core.telemetry import Telemetry
from mentat.core.embeddings import EmbeddingRegistry
from mentat.probes import get_probe, run_probe
from mentat.probes.base import ProbeResult
from mentat.librarian.engine import Librarian
from mentat.storage.vector_db import LanceDBStorage
from mentat.storage.file_store import LocalFileStore
from mentat.storage.cache import ContentHashCache
from mentat.storage.collections import CollectionStore
from mentat.adaptors import BaseAdaptor

# Load .env at import time so all env vars (including API keys for litellm)
# are available before any LLM/embedding calls.
load_dotenv()


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

    # Instruction model — Phase 2: document-level reading guide (use a smart model)
    instruction_model: str = ""
    instruction_api_key: str = ""
    instruction_api_base: str = ""

    # Embedding model
    embedding_provider: str = ""
    embedding_model: str = ""
    embedding_api_key: str = ""
    embedding_api_base: str = ""

    def __post_init__(self):
        self.db_path = self.db_path or os.getenv("MENTAT_DB_PATH", "./mentat_db")
        self.storage_dir = self.storage_dir or os.getenv(
            "MENTAT_STORAGE_DIR", "./mentat_files"
        )

        # Phase 1: Summary model (fast/cheap — bulk chunk summarisation)
        self.summary_model = self.summary_model or os.getenv(
            "MENTAT_SUMMARY_MODEL", "gpt-4o-mini"
        )
        self.summary_api_key = self.summary_api_key or os.getenv(
            "MENTAT_SUMMARY_API_KEY", ""
        )
        self.summary_api_base = self.summary_api_base or os.getenv(
            "MENTAT_SUMMARY_API_BASE", ""
        )

        # Phase 2: Instruction model (smart — document-level guide)
        # Falls back to summary_model if not set.
        self.instruction_model = self.instruction_model or os.getenv(
            "MENTAT_INSTRUCTION_MODEL", ""
        ) or self.summary_model
        self.instruction_api_key = self.instruction_api_key or os.getenv(
            "MENTAT_INSTRUCTION_API_KEY", ""
        ) or self.summary_api_key
        self.instruction_api_base = self.instruction_api_base or os.getenv(
            "MENTAT_INSTRUCTION_API_BASE", ""
        ) or self.summary_api_base

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
        self.config = config or MentatConfig()
        self.logger = logging.getLogger("mentat")

        # Layer 1: Storage (Haystack)
        self.storage = LanceDBStorage(db_path=self.config.db_path)
        self.file_store = LocalFileStore(storage_dir=self.config.storage_dir)
        self.cache = ContentHashCache(cache_dir=self.config.db_path)

        # Layer 3: Librarian
        self.librarian = Librarian(
            model=self.config.instruction_model,
            summary_model=self.config.summary_model,
            api_key=self.config.instruction_api_key or None,
            api_base=self.config.instruction_api_base or None,
            summary_api_key=self.config.summary_api_key or None,
            summary_api_base=self.config.summary_api_base or None,
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

    def register_adaptor(self, adaptor: BaseAdaptor):
        """Register an adaptor for lifecycle hooks."""
        self._adaptors.append(adaptor)

    async def add(
        self,
        path: str,
        force: bool = False,
        summarize: bool = False,
        use_llm_instructions: bool = False,
    ) -> str:
        """Index a file: probe → summarise chunks → generate guide → embed → store.

        Pipeline:
          1. Probe (Layer 2) — extract semantic fingerprint, no LLM.
          2. Chunk summarisation (Librarian phase 1) — LLM summarises each chunk.
             **SKIPPED by default** (set summarize=True to enable).
          3. Instruction generation (Librarian phase 2) — Uses template by default
             (set use_llm_instructions=True for LLM-based generation).
          4. Embed chunks into vector DB.
          5. Store raw file for future detailed access.

        Args:
            path: File path to index
            force: Re-index even if content hash exists in cache
            summarize: Enable LLM-based chunk summarization (default: False, lazy)
            use_llm_instructions: Use LLM for instruction generation (default: False, template-based)

        Returns the document ID.  Skips processing if an identical file
        (by content hash) was already indexed, unless *force=True*.
        """
        # Check content hash cache
        if not force:
            cached_id = self.cache.get(path)
            if cached_id:
                self.logger.info(f"Cache hit for {path} -> {cached_id}, skipping.")
                print(
                    f"[Cache] Already indexed as {cached_id}. Use force=True to re-index."
                )
                return cached_id

        doc_id = str(uuid.uuid4())
        filename = Path(path).name
        self.logger.info(f"Adding file: {path} (ID: {doc_id})")

        # Layer 2: Probe — extract skeleton (no LLM)
        with Telemetry.time_it(doc_id, "probe"):
            probe_result = run_probe(path)
            probe_result.doc_id = doc_id

        Telemetry.record_chunks(doc_id, len(probe_result.chunks))

        # Record fast mode (template + no summarization)
        fast_mode = not summarize and not use_llm_instructions
        if fast_mode:
            stats = Telemetry.get_stats(doc_id)
            if stats:
                stats.fast_mode = True

        # ── Optimized pipeline ───────────────────────────────────────
        # By default, skip LLM summarization and use template-based instructions.
        # This reduces indexing time from ~60s to ~10s while preserving structure.
        #
        # Default (fast):  Probe → Embedding → Template Instructions → Store
        # Full (slow):     Probe → [Summarize → LLM Instructions] + Embedding → Store

        async def _embed():
            with Telemetry.time_it(doc_id, "embedding"):
                embed_texts = []
                for chunk in probe_result.chunks:
                    text = chunk.content
                    if chunk.section:
                        text = f"[{chunk.section}] {text}"
                    embed_texts.append(text)
                if not embed_texts:
                    return []
                return await self.embeddings.embed_batch(embed_texts)

        # Start embedding (always needed)
        self.logger.info(f"Starting embedding for {len(probe_result.chunks)} chunks...")
        embed_task = asyncio.create_task(_embed())

        # Conditional summarization
        if summarize:
            self.logger.info(f"Starting summarization for {len(probe_result.chunks)} chunks...")
            async def _summarize():
                with Telemetry.time_it(doc_id, "summarize"):
                    return await self.librarian.summarize_chunks(probe_result)

            chunk_summaries = await _summarize()
            self.logger.info(f"Summarization complete")
        else:
            # Lazy: store empty summaries, can be generated later on-demand
            chunk_summaries = [""] * len(probe_result.chunks)

        # Instruction generation: template (fast) or LLM (smart)
        if use_llm_instructions:
            self.logger.info("Generating instructions with LLM...")
            async def _generate_guide():
                with Telemetry.time_it(doc_id, "librarian"):
                    intro, instr, tokens = await self.librarian.generate_guide(
                        probe_result, chunk_summaries=chunk_summaries if summarize else None
                    )
                    Telemetry.record_tokens(doc_id, tokens)
                    return intro, instr

            brief_intro, instructions = await _generate_guide()
            self.logger.info("LLM instruction generation complete")
        else:
            # Template-based (no LLM, no tokens)
            with Telemetry.time_it(doc_id, "librarian"):
                brief_intro, instructions = self.librarian.generate_guide_template(
                    probe_result
                )

        # Wait for embedding to complete
        self.logger.info("Waiting for embedding to complete...")
        vectors = await embed_task
        self.logger.info(f"Embedding complete, got {len(vectors)} vectors")

        # ── Store results ────────────────────────────────────────────

        self.storage.add_stub(
            doc_id=doc_id,
            filename=filename,
            brief_intro=brief_intro,
            instruction=instructions,
            probe_json=json.dumps(probe_result.model_dump(), default=str),
        )

        chunk_records = []
        for chunk, summary, vector in zip(
            probe_result.chunks, chunk_summaries, vectors
        ):
            chunk_records.append(
                {
                    "chunk_id": f"{doc_id}_{chunk.index}",
                    "doc_id": doc_id,
                    "filename": filename,
                    "content": chunk.content,
                    "summary": summary,
                    "section": chunk.section or "",
                    "chunk_index": chunk.index,
                    "vector": vector,
                }
            )

        if chunk_records:
            self.logger.info(f"Storing {len(chunk_records)} chunks in vector DB...")
            self.storage.add_chunks(chunk_records)

        # Raw file storage — kept for future detailed access
        self.file_store.save(path, doc_id)

        # Telemetry
        original_size = os.path.getsize(path)
        stub_size = len(brief_intro.encode("utf-8"))
        savings = 1.0 - (stub_size / max(1, original_size))
        Telemetry.record_savings(doc_id, savings)

        # Adaptor hooks
        for adaptor in self._adaptors:
            adaptor.on_document_indexed(
                doc_id, {"filename": filename, "brief_intro": brief_intro}
            )

        # Record in content hash cache
        self.cache.put(path, doc_id)

        self.logger.info(f"Successfully indexed {filename} (ID: {doc_id})")
        print(Telemetry.format_stats(doc_id))
        return doc_id

    async def search(
        self,
        query: str,
        top_k: int = 5,
        hybrid: bool = False,
        doc_ids: Optional[List[str]] = None,
    ) -> List[MentatResult]:
        """Search for relevant chunks and return results with instructions.

        Each result includes the chunk content, its section context,
        and the document-level brief_intro + instructions from the Librarian.

        Args:
            doc_ids: If provided, restrict search to these documents only.
        """
        # Transform query via adaptors
        for adaptor in self._adaptors:
            query = adaptor.transform_query(query)

        query_vector = await self.embeddings.embed(query)
        raw_results = self.storage.search(
            query_vector, query, limit=top_k, use_hybrid=hybrid, doc_ids=doc_ids
        )

        results = []
        # Cache for document stubs (avoid repeated lookups)
        stub_cache: Dict[str, Dict] = {}

        for r in raw_results:
            doc_id = r.get("doc_id", "")

            # Fetch document-level info
            if doc_id not in stub_cache:
                stub = self.storage.get_stub(doc_id)
                stub_cache[doc_id] = stub or {}

            stub = stub_cache[doc_id]

            results.append(
                MentatResult(
                    doc_id=doc_id,
                    chunk_id=r.get("chunk_id", ""),
                    filename=r.get("filename", ""),
                    section=r.get("section"),
                    content=r.get("content", ""),
                    summary=r.get("summary", ""),
                    brief_intro=stub.get("brief_intro", ""),
                    instructions=stub.get("instruction", ""),
                    score=r.get("_distance", 0.0),
                )
            )

        # Adaptor hooks
        result_dicts = [r.model_dump() for r in results]
        for adaptor in self._adaptors:
            result_dicts = adaptor.on_search_results(query, result_dicts)

        return results

    async def inspect(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full probe results, instructions, and chunk summaries."""
        stub = self.storage.get_stub(doc_id)
        if not stub:
            return None

        result: Dict[str, Any] = {
            "doc_id": doc_id,
            "filename": stub.get("filename"),
            "brief_intro": stub.get("brief_intro"),
            "instruction": stub.get("instruction"),
        }

        # Parse probe JSON if available
        probe_json = stub.get("probe_json", "")
        if probe_json:
            try:
                result["probe"] = json.loads(probe_json)
            except json.JSONDecodeError:
                pass

        # Fetch stored chunk summaries
        chunk_rows = self.storage.get_chunks_by_doc(doc_id)
        if chunk_rows:
            result["chunk_summaries"] = [
                {
                    "index": r.get("chunk_index", 0),
                    "section": r.get("section", ""),
                    "summary": r.get("summary", ""),
                }
                for r in chunk_rows
            ]

        return result

    def collection(self, name: str) -> "Collection":
        """Get a Collection handle for scoped add/search operations."""
        return Collection(name, self)

    def list_collections(self) -> List[str]:
        """Return all collection names."""
        return self.collections_store.list_collections()

    async def summarize_doc(self, doc_id: str) -> bool:
        """Generate summaries for a document's chunks (for lazy/background summarization).

        This can be called after indexing to generate summaries on-demand.
        Returns True if summaries were generated, False if doc not found or already summarized.
        """
        # Get stored chunks
        chunk_rows = self.storage.get_chunks_by_doc(doc_id)
        if not chunk_rows:
            return False

        # Check if summaries already exist (non-empty)
        if all(row.get("summary", "") for row in chunk_rows):
            self.logger.info(f"Document {doc_id} already has summaries")
            return False

        # Get probe result from stub
        stub = self.storage.get_stub(doc_id)
        if not stub:
            return False

        probe_json = stub.get("probe_json", "")
        if not probe_json:
            return False

        probe_result = ProbeResult.model_validate_json(probe_json)

        # Generate summaries
        self.logger.info(f"Generating summaries for {doc_id}...")
        chunk_summaries = await self.librarian.summarize_chunks(probe_result)

        # Update chunks in storage
        # LanceDB requires delete + re-add for updates
        # TODO: This is inefficient, but LanceDB doesn't have UPDATE yet
        # For now, we just log that summaries were generated but don't update
        # In production, you'd want to implement chunk updates or use a DB that supports UPDATE

        self.logger.warning(
            f"Summaries generated for {doc_id} but not persisted (LanceDB limitation). "
            "Consider re-indexing with --summarize flag."
        )

        return True

    def stats(self) -> Dict[str, Any]:
        """Return system statistics."""
        return {
            "docs_indexed": self.storage.count_docs(),
            "chunks_stored": self.storage.count_chunks(),
            "cached_hashes": len(self.cache),
            "storage_size_bytes": self.file_store.total_size(),
            "collections": len(self.collections_store.list_collections()),
        }


class Collection:
    """A named group of documents — scoped view over shared storage.

    Acts as a thin wrapper around Mentat. Documents are indexed into the
    shared store; the collection just holds doc_id references.
    """

    def __init__(self, name: str, mentat: Mentat):
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
        )
        self._store.add_doc(self.name, doc_id)
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
