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


__all__ = [
    "add",
    "search",
    "inspect",
    "probe",
    "stats",
    "collection",
    "collections",
    "configure",
    "Mentat",
    "MentatConfig",
    "MentatResult",
    "Collection",
    "ProbeResult",
]
