"""
Mentat: Pure logic. Strategic retrieval.

Usage:
    import mentat

    doc_id = await mentat.add("paper.pdf")
    results = await mentat.search("What algorithm is used?")
    probe_result = mentat.probe("data.csv")
"""

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
from mentat.probes.base import ProbeResult, TopicInfo, StructureInfo, Chunk
from mentat.probes import run_probe
from mentat.skill import get_tool_schemas, get_system_prompt, export_skill

# ── Service layer (shared by CLI, server, SDK) ───────────────────────
from mentat import service

# ── Module-level convenience API ─────────────────────────────────────
# These delegate to the Mentat singleton so callers can write
# ``await mentat.add(...)`` instead of ``Mentat.get_instance().add(...)``.

_get = Mentat.get_instance


def configure(config: MentatConfig):
    """Configure Mentat with custom settings. Must be called before first use."""
    Mentat.reset()
    Mentat.get_instance(config)


# Indexing
async def add(path, **kw):
    return await _get().add(path, **kw)

async def add_batch(paths, **kw):
    return await _get().add_batch(paths, **kw)

async def add_content(content, filename, **kw):
    return await _get().add_content(content, filename, **kw)

# Search & retrieval
async def search(query, **kw):
    return await _get().search(query, **kw)

async def search_grouped(query, **kw):
    return await _get().search_grouped(query, **kw)

async def inspect(doc_id, **kw):
    return await _get().inspect(doc_id, **kw)

async def get_doc_meta(doc_id):
    return await _get().get_doc_meta(doc_id)

async def read_segment(doc_id, section_path, **kw):
    return await _get().read_segment(doc_id, section_path, **kw)

async def read_structured(path, **kw):
    return await _get().read_structured(path, **kw)

# Probe (no LLM, no storage)
def probe(path):
    return run_probe(path)

# Lifecycle
async def start_processor():
    await _get().start()

async def shutdown():
    await _get().shutdown()

# Status
def get_status(doc_id):
    return _get().get_processing_status(doc_id)

async def wait_for(doc_id, timeout=300):
    return await _get().wait_for_completion(doc_id, timeout=timeout)

def stats():
    return _get().stats()

# Access tracking
async def track_access(path):
    return await _get().track_access(path)

def get_section_heat(doc_id=None, limit=20):
    return _get().get_section_heat(doc_id=doc_id, limit=limit)

# Collections
def collection(name):
    return _get().collection(name)

def collections():
    return _get().list_collections()

def create_collection(name, **kw):
    return _get().collections_store.create(name, **kw)

def get_collection_info(name):
    return _get().collections_store.get(name)

def delete_collection(name):
    return _get().collections_store.delete_collection(name)

def gc_collections():
    return _get().collections_store.gc()
