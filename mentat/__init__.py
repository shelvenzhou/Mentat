"""
Mentat: Strategic Retrieval for LLM agents.

Usage::

    import mentat

    # Index a file
    doc_id = await mentat.add("paper.pdf")

    # Two-step retrieval: discover, then extract
    results = await mentat.search("revenue", toc_only=True)
    segment = await mentat.read_segment(doc_id, "Q3 Highlights")

    # Standard RAG search
    results = await mentat.search("What algorithm is used?")

    # Probe (no LLM, no storage)
    probe_result = mentat.probe("data.csv")
"""

__version__ = "0.2.0"

# ── Core classes & models ────────────────────────────────────────────
from mentat.core.hub import Mentat
from mentat.core.models import (
    MentatConfig,
    MentatResult,
    MentatDocResult,
    ChunkResult,
    Collection,
    BaseAdaptor,
)
from mentat.probes.base import ProbeResult, TopicInfo, StructureInfo, Chunk, TocEntry, Caption
from mentat.probes import run_probe
from mentat.skill import get_tool_schemas, get_system_prompt, export_skill
from mentat.storage.base import BaseVectorStorage
from mentat.storage.filters import MetadataFilter, MetadataFilterSet

# ── Service layer (shared by CLI, server, SDK) ───────────────────────
from mentat import service

# ── Type imports ─────────────────────────────────────────────────────
from typing import Any, Dict, List, Optional

# ── Singleton shorthand ──────────────────────────────────────────────
_get = Mentat.get_instance


# ── Configuration ────────────────────────────────────────────────────


def configure(config: MentatConfig) -> None:
    """Apply custom configuration. Must be called before first use.

    Args:
        config: A ``MentatConfig`` instance with desired settings.
            Any field not set falls back to ``MENTAT_*`` env vars, then defaults.
    """
    Mentat.reset()
    Mentat.get_instance(config)


# ── Indexing ─────────────────────────────────────────────────────────


async def add(
    path: str,
    *,
    force: bool = False,
    summarize: bool = False,
    use_llm_instructions: bool = False,
    wait: bool = False,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> str:
    """Index a file. Returns document ID.

    By default returns immediately after probe + stub storage (~1-3 s).
    Embeddings are processed in the background.

    Args:
        path: File path to index.
        force: Re-index even if content hash exists in cache.
        summarize: Enable LLM-based chunk summarization.
        use_llm_instructions: Use LLM for instruction generation (default: template-based).
        wait: Block until background processing completes.
        source: Origin tag for provenance (e.g. ``"web_fetch"``, ``"composio:gmail"``).
        metadata: Arbitrary key-value metadata stored with the document.
        collection: Add the document to this named collection.

    Returns:
        Document ID (UUID string). ToC is available immediately;
        full vector search becomes available after background processing.
    """
    return await _get().add(
        path,
        force=force,
        summarize=summarize,
        use_llm_instructions=use_llm_instructions,
        wait=wait,
        source=source,
        metadata=metadata,
        collection=collection,
    )


async def add_batch(
    paths: List[str],
    *,
    force: bool = False,
    summarize: bool = False,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Index multiple files with batched embedding (one API call for all chunks).

    Much faster than calling ``add(wait=True)`` in a loop.

    Args:
        paths: List of file paths to index.
        force: Re-index even if already cached.
        summarize: Enable LLM-based chunk summarization.
        source: Origin tag applied to all documents.
        metadata: Metadata applied to all documents.

    Returns:
        List of document IDs in the same order as *paths*.
    """
    return await _get().add_batch(
        paths, force=force, summarize=summarize, source=source, metadata=metadata
    )


async def add_content(
    content: str,
    filename: str,
    *,
    content_type: str = "text/plain",
    force: bool = False,
    summarize: bool = False,
    wait: bool = False,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> str:
    """Index raw text content without a file on disk. Returns document ID.

    Args:
        content: The text content to index.
        filename: Logical filename (used for format detection and display).
        content_type: MIME type hint (e.g. ``"text/html"``, ``"text/markdown"``).
        force: Re-index even if content hash exists.
        summarize: Enable LLM-based chunk summarization.
        wait: Block until background processing completes.
        source: Origin tag for provenance.
        metadata: Arbitrary key-value metadata.
        collection: Add to this named collection.
    """
    return await _get().add_content(
        content,
        filename,
        content_type=content_type,
        force=force,
        summarize=summarize,
        wait=wait,
        source=source,
        metadata=metadata,
        collection=collection,
    )


# ── Search & retrieval ───────────────────────────────────────────────


async def search(
    query: str,
    *,
    top_k: int = 5,
    hybrid: bool = False,
    toc_only: bool = False,
    source: Optional[str] = None,
    with_metadata: Optional[bool] = None,
    collections: Optional[List[str]] = None,
) -> List[MentatResult]:
    """Search for relevant content.

    Args:
        query: Natural language search query.
        top_k: Maximum number of results.
        hybrid: Combine vector similarity with keyword matching.
        toc_only: Return document-level ToC summaries instead of chunk content
            (step 1 of two-step retrieval).
        source: Filter by source tag (exact or glob, e.g. ``"composio:*"``).
        with_metadata: Include ``brief_intro``, ``instructions``, ``toc_entries``.
            Defaults to ``True`` when *toc_only* is set, ``False`` otherwise.
        collections: Restrict search to docs in these collections (OR semantics).
    """
    return await _get().search(
        query,
        top_k=top_k,
        hybrid=hybrid,
        toc_only=toc_only,
        source=source,
        with_metadata=with_metadata,
        collections=collections,
    )


async def search_grouped(
    query: str,
    *,
    top_k: int = 5,
    hybrid: bool = False,
    toc_only: bool = False,
    source: Optional[str] = None,
    with_metadata: Optional[bool] = None,
    collections: Optional[List[str]] = None,
) -> List[MentatDocResult]:
    """Search for relevant content, grouped by document.

    Each document appears once with all matching chunks nested,
    avoiding duplicate metadata. More token-efficient for multi-document queries.

    Args:
        Same as :func:`search`.
    """
    return await _get().search_grouped(
        query,
        top_k=top_k,
        hybrid=hybrid,
        toc_only=toc_only,
        source=source,
        with_metadata=with_metadata,
        collections=collections,
    )


async def inspect(
    doc_id: str,
    *,
    sections: Optional[List[str]] = None,
    full: bool = False,
) -> Optional[dict]:
    """Retrieve document metadata and optionally specific section summaries.

    Lightweight by default (ToC + brief_intro). Pass *sections* for
    specific section chunk summaries, or *full=True* for everything.
    """
    return await _get().inspect(doc_id, sections=sections, full=full)


async def get_doc_meta(doc_id: str) -> Optional[dict]:
    """Get lightweight metadata for a document.

    Returns brief_intro, instructions, toc_entries, source, metadata,
    and processing status. Returns ``None`` if not found.
    """
    return await _get().get_doc_meta(doc_id)


async def read_segment(
    doc_id: str,
    section_path: str,
    *,
    include_summary: bool = True,
) -> dict:
    """Read a specific section from an indexed document.

    Step 2 of the two-step retrieval protocol. Use section names
    from ``get_doc_meta``'s ToC. Parent sections automatically include
    all child sections' content.

    Args:
        doc_id: Document ID from search results.
        section_path: Section name to read (case-insensitive match).
        include_summary: Include chunk summaries alongside content.
    """
    return await _get().read_segment(
        doc_id, section_path, include_summary=include_summary
    )


async def read_structured(
    path: str,
    *,
    sections: Optional[List[str]] = None,
    include_content: bool = False,
) -> dict:
    """Return a structured, token-efficient view of a file (ToC + summaries).

    Args:
        path: File path to read.
        sections: If provided, only include these sections.
        include_content: Include full chunk content (default: summaries only).
    """
    return await _get().read_structured(
        path, sections=sections, include_content=include_content
    )


# ── Probe (no LLM, no storage) ──────────────────────────────────────


def probe(path: str) -> ProbeResult:
    """Run probes on a file. Returns structured extraction result.

    Pure local processing (no LLM calls, no storage). Useful for
    inspecting a file's structure before deciding to index it.
    """
    return run_probe(path)


# ── Lifecycle ────────────────────────────────────────────────────────


async def start_processor() -> None:
    """Start the background processing worker.

    Call once on application startup to enable async document processing.
    If not called, documents will be queued but not processed.
    """
    await _get().start()


async def shutdown() -> None:
    """Shutdown the background processor and file watcher gracefully.

    Waits for currently processing tasks to complete, then persists
    heat maps. Call on application shutdown.
    """
    await _get().shutdown()


# ── Status ───────────────────────────────────────────────────────────


def get_status(doc_id: str) -> dict:
    """Get processing status for a document.

    Returns:
        Dict with keys: ``doc_id``, ``status`` (one of ``"pending"``,
        ``"processing"``, ``"completed"``, ``"failed"``, ``"not_found"``),
        ``submitted_at``, ``error``, ``needs_summarization``.
    """
    return _get().get_processing_status(doc_id)


async def wait_for(doc_id: str, timeout: float = 300) -> bool:
    """Wait for a document's background processing to complete.

    Args:
        doc_id: Document identifier.
        timeout: Maximum wait time in seconds (default: 300).

    Returns:
        ``True`` if processing completed successfully, ``False`` on timeout or failure.
    """
    return await _get().wait_for_completion(doc_id, timeout=timeout)


def stats() -> dict:
    """Return system statistics.

    Returns:
        Dict with keys: ``docs_indexed``, ``chunks_stored``, ``cached_hashes``,
        ``storage_size_bytes``, ``collections``, ``access_tracker``, ``section_heat``.
    """
    return _get().stats()


# ── Access tracking ──────────────────────────────────────────────────


async def track_access(path: str) -> dict:
    """Record a file access event.

    Frequently accessed files are automatically promoted to the hot queue
    and indexed with LLM summarization.

    Returns:
        Dict with ``path``, ``promoted``, ``is_hot``, ``is_recent``, ``tracker_stats``.
    """
    return await _get().track_access(path)


def get_section_heat(
    doc_id: Optional[str] = None,
    limit: int = 20,
) -> list:
    """Return hottest sections by decayed score.

    Args:
        doc_id: If provided, filter to this document only.
        limit: Maximum number of sections to return.
    """
    return _get().get_section_heat(doc_id=doc_id, limit=limit)


# ── Collections ──────────────────────────────────────────────────────


def collection(name: str) -> Collection:
    """Get a named collection for scoped add/search operations.

    The returned :class:`Collection` object provides ``add()``, ``search()``,
    ``remove()``, ``list_docs()``, and ``delete()`` methods.
    """
    return _get().collection(name)


def collections() -> List[str]:
    """List all collection names."""
    return _get().list_collections()


def create_collection(
    name: str,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    watch_paths: Optional[List[str]] = None,
    watch_ignore: Optional[List[str]] = None,
    auto_add_sources: Optional[List[str]] = None,
) -> dict:
    """Create or update a collection with config.

    Args:
        name: Collection name.
        metadata: Arbitrary metadata (supports ``ttl`` key for auto-GC).
        watch_paths: Glob patterns for file watching.
        watch_ignore: Glob patterns to exclude from watching.
        auto_add_sources: Source patterns that auto-route documents to this collection.

    Returns:
        The full collection record dict.
    """
    return _get().collections_store.create(
        name,
        metadata=metadata,
        watch_paths=watch_paths,
        watch_ignore=watch_ignore,
        auto_add_sources=auto_add_sources,
    )


def get_collection_info(name: str) -> Optional[dict]:
    """Get full collection record, or ``None`` if not found."""
    return _get().collections_store.get(name)


def delete_collection(name: str) -> bool:
    """Delete a collection (does NOT delete the underlying documents)."""
    return _get().collections_store.delete_collection(name)


def gc_collections() -> List[str]:
    """Remove expired collections (based on ``metadata.ttl``). Returns deleted names."""
    return _get().collections_store.gc()


# ── Document management ──────────────────────────────────────────────


def list_docs(source: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all indexed documents.

    Args:
        source: If provided, filter to documents with this source tag.

    Returns:
        List of stub dicts with keys like ``id``, ``filename``, ``source``, etc.
    """
    return service.list_docs(source=source)


def delete(doc_id: str) -> None:
    """Delete a document and all its chunks from storage.

    Args:
        doc_id: Document ID to delete.
    """
    m = _get()
    m.storage.delete_doc(doc_id)
    m.cache.remove(doc_id)
    m.path_index.remove(doc_id)


def delete_by_session_id(session_id: str, collection: str | None = None) -> list[str]:
    """Delete all documents associated with a session_id.

    Args:
        session_id: Session identifier to purge.
        collection: If provided, also remove the doc_ids from this collection.

    Returns:
        List of doc_ids that were removed.
    """
    m = _get()
    removed = m.storage.delete_docs_by_session_id(session_id)
    for doc_id in removed:
        m.cache.remove(doc_id)
        m.path_index.remove(doc_id)
        if collection:
            m.collections_store.remove_doc(collection, doc_id)
    return removed


# ── Skill integration ────────────────────────────────────────────────
# get_tool_schemas, get_system_prompt, export_skill are imported at top level.


# ── Adaptor registration ─────────────────────────────────────────────


def register_adaptor(adaptor: BaseAdaptor) -> None:
    """Register an adaptor for lifecycle hooks.

    Adaptors receive callbacks on document indexing and search results,
    and can transform queries before search.
    """
    _get().register_adaptor(adaptor)


# ── Public API ───────────────────────────────────────────────────────

__all__ = [
    # Version
    "__version__",
    # Core classes
    "Mentat",
    "MentatConfig",
    # Result models
    "MentatResult",
    "MentatDocResult",
    "ChunkResult",
    # Probe models
    "ProbeResult",
    "TopicInfo",
    "StructureInfo",
    "Chunk",
    "TocEntry",
    "Caption",
    # Extensibility
    "BaseAdaptor",
    "BaseVectorStorage",
    "MetadataFilter",
    "MetadataFilterSet",
    "Collection",
    # Indexing
    "add",
    "add_batch",
    "add_content",
    # Search & retrieval
    "search",
    "search_grouped",
    "inspect",
    "get_doc_meta",
    "read_segment",
    "read_structured",
    # Probe
    "probe",
    # Lifecycle
    "start_processor",
    "shutdown",
    "configure",
    # Status
    "get_status",
    "wait_for",
    "stats",
    # Access tracking
    "track_access",
    "get_section_heat",
    # Collections
    "collection",
    "collections",
    "create_collection",
    "get_collection_info",
    "delete_collection",
    "gc_collections",
    # Document management
    "list_docs",
    "delete",
    # Skill integration
    "get_tool_schemas",
    "get_system_prompt",
    "export_skill",
    # Adaptor registration
    "register_adaptor",
]
