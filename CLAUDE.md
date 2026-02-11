# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mentat is a next-generation Agentic RAG system (Python 3.10+) that transforms "Content Retrieval" into "Strategy Retrieval." Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract structure/metadata, then a Librarian LLM generates actionable reading guides.

## Development Setup

```bash
# Package manager: uv
uv sync          # Install all dependencies

# CLI entry point
mentat --debug [COMMAND]
```

No test framework or linting tools are configured yet.

## CLI Commands

```bash
mentat probe <file_paths> [--format rich|json]        # Run probes (no LLM, no storage)
mentat index <path> [--force] [-c <collection>]       # Index file/directory
mentat search <query> [--top-k 5] [-c <collection>]   # Search (optionally scoped)
mentat inspect <doc_id>                                # Show probe results + instructions
mentat stats                                           # System statistics
mentat collection list                                 # List all collections
mentat collection show <name>                          # Show docs in a collection
mentat collection delete <name>                        # Delete a collection (not docs)
mentat collection remove <name> <doc_id>               # Remove doc from collection
```

## Architecture

Three-layer design where each layer feeds into the next:

**Layer 1 — Haystack (Storage):** `mentat/storage/`
- `LanceDBStorage` — vector DB with separate tables for document stubs and chunks
- `LocalFileStore` — raw file storage
- `ContentHashCache` — SHA-256 deduplication to skip re-indexing identical files
- `CollectionStore` — named groups of doc_id references (JSON-backed, no vector duplication)

**Layer 2 — Probes (Statistical Extraction):** `mentat/probes/`
- `BaseProbe` ABC with `can_handle()` and `run()` → returns `ProbeResult`
- Registry in `__init__.py` — probes tried in order, first match wins
- Implementations: PDF (pymupdf font analysis), CSV (pandas), JSON (schema inference), Python (tree-sitter AST), Markdown (regex), Web/HTML (trafilatura)

**Layer 3 — Librarian (Instruction Generation):** `mentat/librarian/`
- Uses `litellm` for LLM calls (supports OpenAI, Claude, Gemini, Ollama, etc.)
- Takes **only** `ProbeResult` as input — never reads raw files
- Generates brief intro + actionable instructions ("Reading Guides")

**Orchestrator:** `mentat/core/hub.py`
- `Mentat` class — singleton via `get_instance()`, reset with `reset()`
- `Collection` class — thin wrapper for scoped add/search over a named doc group
- `MentatConfig` dataclass — db_path, models, vector dimensions
- Pipeline: `add()` runs probe → librarian → embed → store

**Public API:** `mentat/__init__.py`
- Module-level functions: `add()`, `search()`, `probe()`, `inspect()`, `stats()`, `collection()`, `collections()`, `configure()`
- `add()`, `search()`, `inspect()` are async; `probe()`, `stats()`, `collection()`, `collections()` are sync

## Key Patterns

- **Pydantic v2** for all data models (`ProbeResult`, `TopicInfo`, `StructureInfo`, `Chunk`, `MentatResult`)
- **Async/await** for all LLM and embedding calls via `litellm`
- **Plugin registry** for probes — add new format support by implementing `BaseProbe` and registering in `mentat/probes/__init__.py`
- **Adaptor hooks** (`mentat/adaptors/`) — `BaseAdaptor` interface with `on_document_indexed`, `on_search_results`, `transform_query`
- **Collections** — named doc groups for scoped search; shared storage with doc_id references (no vector duplication); LanceDB `WHERE doc_id IN (...)` pre-filtering with BTREE scalar index
- **Telemetry** (`mentat/core/telemetry.py`) — context-manager-based timing, tracks probe/librarian time, tokens, and context savings
