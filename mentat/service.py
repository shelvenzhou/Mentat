"""Shared service functions for CLI, server, and future SDK.

Each function takes explicit arguments (no request objects) and returns
plain dicts or model instances. The CLI formats them for terminal output;
the server returns them as JSON.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from mentat.core.hub import Mentat
from mentat.storage.filters import MetadataFilter, MetadataFilterSet


def _build_filters(metadata_filter: Optional[Dict[str, Any]]) -> Optional[MetadataFilterSet]:
    """Convert a dict to a MetadataFilterSet for LanceDB.

    Simple form (equality):  ``{"session_id": "abc"}``
    Extended form (any op):  ``{"session_id": {"op": "neq", "value": "abc"}}``
    """
    if not metadata_filter:
        return None
    filters: list[MetadataFilter] = []
    for k, v in metadata_filter.items():
        if isinstance(v, dict) and "op" in v:
            filters.append(MetadataFilter(field=k, op=v["op"], value=v["value"]))
        else:
            filters.append(MetadataFilter(field=k, op="eq", value=v))
    return MetadataFilterSet(filters=filters)


def get_mentat() -> Mentat:
    """Return the Mentat singleton."""
    return Mentat.get_instance()


def resolve_doc_id(prefix: str) -> str:
    """Resolve a doc ID prefix to a full ID.

    Raises:
        ValueError: If the prefix is ambiguous (matches multiple docs).
        KeyError: If no document matches the prefix.
    """
    m = get_mentat()
    try:
        full_id = m.storage.resolve_doc_id(prefix)
    except ValueError:
        raise
    if full_id is None:
        raise KeyError(f"Document not found: {prefix}")
    return full_id


def resolve_collections(
    collection: Optional[str] = None,
    collections: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Normalize collection/collections into a single list (or None)."""
    colls = list(collections or [])
    if collection and collection not in colls:
        colls.append(collection)
    return colls or None


async def index_file(
    path: str,
    force: bool = False,
    summarize: bool = False,
    wait: bool = False,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Index a file. Returns {"doc_id": ..., "status": ...}."""
    m = get_mentat()
    resolved = str(Path(path).resolve())
    doc_id = await m.add(
        resolved,
        force=force,
        summarize=summarize,
        wait=wait,
        source=source,
        metadata=metadata,
        collection=collection,
    )
    status = m.get_processing_status(doc_id)
    return {"doc_id": doc_id, "status": status.get("status", "unknown")}


async def index_content(
    content: str,
    filename: str,
    content_type: str = "text/plain",
    force: bool = False,
    summarize: bool = False,
    wait: bool = False,
    source: str = "",
    metadata: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Index raw content. Returns {"doc_id": ..., "status": ...}."""
    m = get_mentat()
    doc_id = await m.add_content(
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
    status = m.get_processing_status(doc_id)
    return {"doc_id": doc_id, "status": status.get("status", "unknown")}


async def search_docs(
    query: str,
    top_k: int = 5,
    hybrid: bool = False,
    toc_only: bool = False,
    source: Optional[str] = None,
    with_metadata: Optional[bool] = None,
    collection: Optional[str] = None,
    collections: Optional[List[str]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search for documents. Returns list of result dicts.

    Args:
        metadata_filter: Simple key-value equality filters on chunk metadata
            columns, e.g. ``{"session_id": "abc"}``.
    """
    m = get_mentat()
    colls = resolve_collections(collection, collections)
    filters = _build_filters(metadata_filter)
    results = await m.search(
        query,
        top_k=top_k,
        hybrid=hybrid,
        toc_only=toc_only,
        source=source,
        with_metadata=with_metadata,
        collections=colls,
        filters=filters,
    )
    return [r.model_dump() for r in results]


async def search_grouped(
    query: str,
    top_k: int = 5,
    hybrid: bool = False,
    toc_only: bool = False,
    source: Optional[str] = None,
    with_metadata: Optional[bool] = None,
    collection: Optional[str] = None,
    collections: Optional[List[str]] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Search for documents, grouped by doc. Returns list of result dicts."""
    m = get_mentat()
    colls = resolve_collections(collection, collections)
    filters = _build_filters(metadata_filter)
    results = await m.search_grouped(
        query,
        top_k=top_k,
        hybrid=hybrid,
        toc_only=toc_only,
        source=source,
        with_metadata=with_metadata,
        collections=colls,
        filters=filters,
    )
    return [r.model_dump() for r in results]


def list_docs(source: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all indexed documents, optionally filtered by source."""
    m = get_mentat()
    docs = m.storage.list_docs()
    if source:
        docs = [d for d in docs if d.get("source") == source]
    return docs


def get_stats() -> Dict[str, Any]:
    """Return system statistics."""
    return get_mentat().stats()
