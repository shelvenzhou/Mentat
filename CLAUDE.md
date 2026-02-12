# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mentat is a next-generation Agentic RAG system (Python 3.10+) that transforms "Content Retrieval" into "Strategy Retrieval." Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** (hierarchy + metadata + anchors + snippets), then a Librarian LLM generates actionable reading guides. Small files (< 1000 tokens) bypass skeleton extraction and return full content directly.

## Development Setup

```bash
# Package manager: uv
uv sync          # Install all dependencies

# CLI entry point (preferred way to run)
uv run python -m mentat.cli [COMMAND]

# Or after install:
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

**Layer 2 — Probes (Semantic Fingerprinting):** `mentat/probes/`
- `BaseProbe` ABC with `can_handle()` and `run()` → returns `ProbeResult`
- Registry in `__init__.py` — 13 probes tried in order, first match wins
- Shared utilities in `_utils.py` — `estimate_tokens`, `should_bypass`, `extract_preview`, `safe_read_text`
- `TocEntry` fields: `level`, `title`, `page`, `preview` (first sentence), `annotation` (structural features)
- Probes:
  - **PDF** (`pdf_probe.py`) — pymupdf font analysis, native + inferred ToC, per-page chunks
  - **Image** (`image_probe.py`) — Pillow; dimensions, format, EXIF (camera, GPS, date). Optional dep.
  - **Word** (`docx_probe.py`) — python-docx; heading hierarchy, tables, metadata, section chunks. Optional dep.
  - **PowerPoint** (`pptx_probe.py`) — python-pptx; slide titles, bullets, notes, image/table counts. Optional dep.
  - **Calendar** (`calendar_probe.py`) — icalendar; events, recurrence, attendees, time range. Optional dep.
  - **Archive** (`archive_probe.py`) — stdlib zipfile/tarfile; directory tree, file type distribution, size stats
  - **CSV** (`csv_probe.py`) — pandas; column types, cardinality, null rates, representative rows
  - **JSON** (`json_probe.py`) — stdlib; depth-limited schema tree, per-key chunks, value previews
  - **Config** (`config_probe.py`) — pyyaml/tomli/configparser; key hierarchy, value types (.yaml/.toml/.ini/.conf/.cfg)
  - **Code** (`code_probe.py`) — tree-sitter; Python + JS + TS; imports, classes, functions, signatures, docstrings
  - **Log** (`log_probe.py`) — regex; time range, error level stats, format detection, keywords (.log)
  - **Markdown** (`markdown_probe.py`) — regex; heading hierarchy with preview/annotation, section-aware chunks
  - **Web/HTML** (`web_probe.py`) — trafilatura + regex; heading structure, meta tags, semantic elements

**Layer 3 — Librarian (Instruction Generation):** `mentat/librarian/`
- Uses `litellm` for LLM calls (supports OpenAI, Claude, Gemini, Ollama, etc.)
- Takes **only** `ProbeResult` as input — never reads raw files
- Renders ToC with preview/annotation; detects `is_full_content` flag for small files
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

- **Pydantic v2** for all data models (`ProbeResult`, `TopicInfo`, `StructureInfo`, `Chunk`, `TocEntry`, `MentatResult`)
- **Async/await** for all LLM and embedding calls via `litellm`
- **Plugin registry** for probes — add new format support by implementing `BaseProbe` and registering in `mentat/probes/__init__.py`
- **Graceful degradation** — optional probes (Image, DOCX, PPTX, Calendar) use try/except imports; missing deps disable the probe silently
- **Small-file bypass** — files under ~1000 tokens return full content in a single chunk (`stats.is_full_content = True`)
- **Adaptor hooks** (`mentat/adaptors/`) — `BaseAdaptor` interface with `on_document_indexed`, `on_search_results`, `transform_query`
- **Collections** — named doc groups for scoped search; shared storage with doc_id references (no vector duplication); LanceDB `WHERE doc_id IN (...)` pre-filtering with BTREE scalar index
- **Telemetry** (`mentat/core/telemetry.py`) — context-manager-based timing, tracks probe/librarian time, tokens, and context savings
