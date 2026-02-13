# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mentat is a next-generation Agentic RAG system (Python 3.10+) that transforms "Content Retrieval" into "Strategy Retrieval." Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** (hierarchy + metadata + anchors + snippets), then a Librarian LLM generates actionable reading guides. Small files (< 1000 tokens) bypass skeleton extraction and return full content directly.

## Development Setup

```bash
# Package manager: uv
uv sync          # Install all dependencies

# Configuration: copy .env.example → .env and set API keys / model names
cp .env.example .env   # then edit .env

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
- Shared utilities in `_utils.py` — `estimate_tokens`, `should_bypass`, `extract_preview`, `safe_read_text`, `merge_small_chunks`
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
  - **Markdown** (`markdown_probe.py`) — regex; heading hierarchy with preview/annotation, section-aware chunks. Code-fence-aware header detection filters out `#` comments inside fenced blocks.
  - **Web/HTML** (`web_probe.py`) — trafilatura + regex; heading structure, meta tags, semantic elements

**Layer 3 — Librarian (Two-Phase Instruction Generation):** `mentat/librarian/`
- Uses `litellm` for LLM calls (supports OpenAI, Claude, Gemini, Ollama, etc.)
- Takes **only** `ProbeResult` as input — never reads raw files
- **Phase 1 — Chunk Summarisation** (`summary_model`): fast/cheap LLM generates a 1-3 sentence summary for each chunk (batched, concurrent). Small files (`is_full_content`) bypass summarisation. Embeddings are also computed concurrently.
- **Phase 2 — Instruction Generation** (`instruction_model`): smart LLM receives ToC + chunk summaries + statistics and produces:
  - `brief_intro`: 1-2 sentence overview
  - `instructions`: actionable reading guide noting what data is present, what is missing/truncated, and how to access the raw file for details
- Chunk summaries are stored alongside chunks in the vector DB for retrieval
- Original raw files are always kept in `LocalFileStore` for downstream detailed access

**Orchestrator:** `mentat/core/hub.py`
- `Mentat` class — singleton via `get_instance()`, reset with `reset()`
- `Collection` class — thin wrapper for scoped add/search over a named doc group
- `MentatConfig` dataclass — loads from `.env` via `python-dotenv`, with `MENTAT_` prefixed env vars
- Pipeline: `add()` runs probe → summarise chunks → generate guide → embed → store

**Public API:** `mentat/__init__.py`
- Module-level functions: `add()`, `search()`, `probe()`, `inspect()`, `stats()`, `collection()`, `collections()`, `configure()`
- `add()`, `search()`, `inspect()` are async; `probe()`, `stats()`, `collection()`, `collections()` are sync

## Key Patterns

- **Pydantic v2** for all data models (`ProbeResult`, `TopicInfo`, `StructureInfo`, `Chunk`, `TocEntry`, `MentatResult`)
- **Async/await** for all LLM and embedding calls via `litellm`
- **Plugin registry** for probes — add new format support by implementing `BaseProbe` and registering in `mentat/probes/__init__.py`
- **Graceful degradation** — optional probes (Image, DOCX, PPTX, Calendar) use try/except imports; missing deps disable the probe silently
- **Small-file bypass** — files under ~1000 tokens return full content in a single chunk (`stats.is_full_content = True`)
- **Chunk merging** — `merge_small_chunks()` in `_utils.py` post-processes probe output to merge adjacent small chunks (< 300 tokens) while respecting H1/H2 hard boundaries (max merged size: 1200 tokens). Hierarchy levels are communicated via `chunk.metadata["level"]`. Applied by markdown, web, docx, and json probes. ToC entries remain granular; only chunks are merged. Code and PDF probes are exempt (semantically distinct units / page boundaries).
- **Adaptor hooks** (`mentat/adaptors/`) — `BaseAdaptor` interface with `on_document_indexed`, `on_search_results`, `transform_query`
- **Collections** — named doc groups for scoped search; shared storage with doc_id references (no vector duplication); LanceDB `WHERE doc_id IN (...)` pre-filtering with BTREE scalar index
- **Telemetry** (`mentat/core/telemetry.py`) — context-manager-based timing, tracks probe/summarize/librarian time, tokens, and context savings
- **Environment config** — `.env` file loaded at import via `python-dotenv`; `MentatConfig` resolves: explicit arg > env var (`MENTAT_*`) > default. Separate `api_key`/`api_base` for summary model (`MENTAT_SUMMARY_*`), instruction model (`MENTAT_INSTRUCTION_*`), and embedding model (`MENTAT_EMBEDDING_*`); instruction model falls back to summary model when not set. Global provider keys (`OPENAI_API_KEY`, etc.) are read natively by `litellm` as fallback. Vector dimension is auto-detected from the first embedding call (lazy chunks table creation).
