"""
Mentat: Pure logic. Strategic retrieval.

Usage:
    import mentat

    # Index a file
    doc_id = await mentat.add("paper.pdf")

    # Search
    results = await mentat.search("What algorithm is used?")

    # Probe (no LLM, no storage)
    probe_result = mentat.probe("data.csv")

    # Inspect indexed document
    info = await mentat.inspect(doc_id)

    # System stats
    stats = mentat.stats()
"""

from mentat.core.hub import Mentat, MentatConfig, MentatResult, Collection
from mentat.probes import run_probe
from mentat.probes.base import ProbeResult, TopicInfo, StructureInfo, Chunk
from typing import List, Optional


async def add(path: str, force: bool = False, **kwargs) -> str:
    """Index a file. Returns document ID."""
    return await Mentat.get_instance().add(path, force=force, **kwargs)


async def search(
    query: str, top_k: int = 5, hybrid: bool = False
) -> List[MentatResult]:
    """Search for relevant content."""
    return await Mentat.get_instance().search(query, top_k=top_k, hybrid=hybrid)


async def inspect(doc_id: str) -> Optional[dict]:
    """Retrieve full probe + librarian results for a document."""
    return await Mentat.get_instance().inspect(doc_id)


def probe(path: str) -> ProbeResult:
    """Run probes on a file (no LLM, no storage). Returns structured probe result."""
    return run_probe(path)


def stats() -> dict:
    """Return system statistics."""
    return Mentat.get_instance().stats()


def collection(name: str) -> Collection:
    """Get a named collection for scoped add/search operations."""
    return Mentat.get_instance().collection(name)


def collections() -> List[str]:
    """List all collection names."""
    return Mentat.get_instance().list_collections()


def configure(config: MentatConfig):
    """Configure Mentat with custom settings. Must be called before first use."""
    Mentat.reset()
    Mentat.get_instance(config)


# --- Background Processing APIs ---


async def start_processor():
    """Start the background processing worker.

    Should be called once on application startup to enable async document processing.
    If not called, documents will be queued but not processed until start_processor() is invoked.

    Example:
        await mentat.start_processor()
        doc_id = await mentat.add("large.pdf")  # Returns immediately, processes in background
    """
    await Mentat.get_instance().start()


async def shutdown():
    """Shutdown the background processor gracefully.

    Waits for currently processing tasks to complete before stopping.
    Should be called on application shutdown.

    Example:
        await mentat.shutdown()
    """
    await Mentat.get_instance().shutdown()


def get_status(doc_id: str) -> dict:
    """Get processing status for a document.

    Args:
        doc_id: Document identifier

    Returns:
        Status dict with keys:
            - doc_id: Document ID
            - status: "pending" | "processing" | "completed" | "failed" | "not_found"
            - submitted_at: Timestamp when queued (if in queue)
            - error: Error message (if failed)
            - needs_summarization: Whether summarization was requested

    Example:
        status = mentat.get_status(doc_id)
        print(f"Status: {status['status']}")
    """
    return Mentat.get_instance().get_processing_status(doc_id)


async def wait_for(doc_id: str, timeout: float = 300) -> bool:
    """Wait for a document's background processing to complete.

    Args:
        doc_id: Document identifier
        timeout: Maximum wait time in seconds (default: 300 = 5 minutes)

    Returns:
        True if processing completed successfully, False if timeout or failed

    Example:
        doc_id = await mentat.add("file.pdf", wait=False)
        completed = await mentat.wait_for(doc_id)
        if completed:
            print("Processing complete!")
    """
    return await Mentat.get_instance().wait_for_completion(doc_id, timeout=timeout)


__all__ = [
    # Core APIs
    "add",
    "search",
    "inspect",
    "probe",
    "stats",
    "collection",
    "collections",
    "configure",
    # Background processing APIs
    "start_processor",
    "shutdown",
    "get_status",
    "wait_for",
    # Classes
    "Mentat",
    "MentatConfig",
    "MentatResult",
    "Collection",
    "ProbeResult",
]
