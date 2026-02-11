"""
Adaptors for integrating Mentat with external systems.

This package provides the base interface for adaptor implementations.
A future OpenClaw adaptor would implement BaseAdaptor to bridge
OpenClaw's existing RAG/memory system with Mentat's strategy retrieval.
"""

import abc
from typing import Any, Dict, List, Optional


class BaseAdaptor(abc.ABC):
    """Base class for integrating Mentat with external systems.

    An adaptor can hook into Mentat's lifecycle:
    - on_document_indexed: called after a document is indexed
    - on_search_results: called after search, can transform/filter results
    - transform_query: called before search, can rewrite queries

    Example (future OpenClaw adaptor):
        class OpenClawAdaptor(BaseAdaptor):
            def on_document_indexed(self, doc_id, metadata):
                # Sync to OpenClaw's memory store
                ...
            def on_search_results(self, query, results):
                # Merge with OpenClaw's own search results
                ...
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
