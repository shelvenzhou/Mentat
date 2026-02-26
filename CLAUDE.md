# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Mentat is a next-generation Agentic RAG system (Python 3.10+) that transforms "Content Retrieval" into "Strategy Retrieval." Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** (hierarchy + metadata + anchors + snippets), then generates template-based actionable reading guides. Small files (< 1000 tokens) bypass skeleton extraction and return full content directly.

## Development Commands

```bash
# Package manager: uv
uv sync                              # Install all dependencies

# Configuration
cp .env.example .env                 # Then edit .env with API keys / model names

# CLI (preferred way to run)
uv run python -m mentat.cli [COMMAND]
mentat --debug [COMMAND]             # After install

# FastAPI server (port 7832)
mentat serve

# Tests (all mocked, no API keys needed; pytest-asyncio with asyncio_mode=auto)
uv run pytest tests/ -v              # Full suite
uv run pytest tests/test_smoke.py -v # End-to-end smoke test
uv run pytest tests/test_queue.py -v # Single test file
uv run pytest tests/test_smoke.py::test_probe_markdown -v  # Single test
```

## Architecture

Three-layer unidirectional pipeline: **Probes → Librarian → Storage**

### Layer 1 — Probes (`mentat/probes/`)
Semantic fingerprinting — no LLM, pure extraction. Each probe implements `BaseProbe` ABC (`can_handle()` + `run()` → `ProbeResult`). Registry in `__init__.py` — 13 probes tried in order, first match wins. Optional-dep probes (Image, DOCX, PPTX, Calendar) degrade gracefully via try/except import.

Key utilities in `_utils.py`: `estimate_tokens` (1 token ≈ 3 chars), `normalize_chunk_sizes` (merges adjacent small chunks <300 tokens, respects H1/H2 boundaries, max merged 1200 tokens), `should_bypass` (<1000 token files return full content).

Format-specific reading guide templates live in `instruction_templates.py`.

### Layer 2 — Librarian (`mentat/librarian/engine.py`)
Uses `litellm` for all LLM calls. Takes only `ProbeResult` as input — never reads raw files.
- **Phase 1 — Chunk Summarisation** (optional): LLM generates 1-3 sentence summaries per chunk (batched, concurrent). Small files bypass this.
- **Phase 2 — Instruction Generation** (template-based, no LLM): produces `brief_intro` + `instructions` from ToC + statistics.

### Layer 3 — Storage (`mentat/storage/`)
- `LanceDBStorage` (`vector_db.py`) — separate tables for document stubs and chunks. Lazy vector table creation (dimension auto-detected from first embedding). BTREE scalar index on `doc_id` for collection filtering.
- `LocalFileStore` (`file_store.py`) — raw file copies for downstream access.
- `ContentHashCache` (`cache.py`) — SHA-256 deduplication (JSON-backed).
- `CollectionStore` (`collections.py`) — named doc groups as JSON references (no vector duplication).

### Orchestrator (`mentat/core/hub.py`)
`Mentat` class — singleton via `get_instance()`, reset with `reset()`. Houses `MentatConfig` dataclass (loads `.env` via python-dotenv, `MENTAT_*` env vars, precedence: explicit arg > env var > default).

**Async processing pipeline** (default): `add()` returns in ~1-3s after probe + stub storage, then queues background embeddings/summarization. Legacy sync: `add(wait=True)` blocks until complete.

### Background Queue (`mentat/core/queue.py`)
`ProcessingQueue` + `BackgroundProcessor` — in-memory priority queue (transient, lost on restart). Priority boosting: documents queried before processing completes get +10 priority. Concurrency: `max_concurrent_tasks` (default 5, via `MENTAT_MAX_CONCURRENT_TASKS`).

### Other Core Modules
- `core/embeddings.py` — `LiteLLMEmbedding` provider with batching. Oversized chunks split with 500-char overlap.
- `core/access_tracker.py` — two-layer FIFO: recent (LRU) → hot (≥2 accesses). Promotion callback triggers on-demand summarization. Persistent heat map via `heat_map.json` (debounced writes, loaded on init).
- `core/telemetry.py` — context-manager timing for probe/summarize/librarian phases, token savings tracking.
- `adaptors/__init__.py` — `BaseAdaptor` ABC with `on_document_indexed`, `on_search_results`, `transform_query` hooks.
- `server.py` — FastAPI HTTP server with endpoints for index, search, inspect, status, probe, read-segment, skill, collections, access tracking.
- `skill.py` — Skill Integration Layer: OpenAI function calling tool schemas + system prompt fragment for agent two-step retrieval protocol. `export_skill()` returns combined payload.

### Public API (`mentat/__init__.py`)
Module-level async functions: `add()`, `add_batch()`, `add_content()`, `search()`, `inspect()`, `read_structured()`, `read_segment()`, `track_access()`, `start_processor()`, `shutdown()`, `wait_for()`. Sync functions: `probe()`, `stats()`, `collection()`, `collections()`, `get_status()`, `configure()`, `get_tool_schemas()`, `get_system_prompt()`, `export_skill()`.

## CLI Commands

```bash
mentat probe <file_paths> [--format rich|json]        # Run probes (no LLM, no storage)
mentat index <path> [--force] [-c <collection>] [--summarize] [--llm-instructions] [--wait] [-j N]
mentat status <doc_id>                                 # Check processing status
mentat search <query> [--top-k 5] [--hybrid] [-c <collection>] [--toc-only]
mentat segment <doc_id> <section>                      # Read specific section (two-step protocol step 2)
mentat inspect <doc_id>                                # Show probe results + instructions
mentat stats                                           # System statistics
mentat collection list|show|delete|remove              # Collection management
mentat skill [--format json|prompt]                     # Export agent tool schemas + system prompt
mentat serve                                           # Start FastAPI server (port 7832)
```

## Two-Step Retrieval Protocol (Agent Integration)

Mentat implements a Probe→Fetch protocol for token-efficient agent memory access:

1. **Step 1 — Discover**: `search(query, toc_only=True)` returns document summaries + ToC entries (no chunk content). Agent sees ~100-200 tokens per doc.
2. **Step 2 — Read**: `read_segment(doc_id, section_path)` fetches specific section content by doc_id + section name from step 1.

The `skill.py` module exports OpenAI function calling tool schemas and a system prompt fragment that teaches agents this protocol. Use `mentat skill` CLI or `GET /skill` endpoint.

## Key Patterns

- **Pydantic v2** for all data models (`ProbeResult`, `TopicInfo`, `StructureInfo`, `Chunk`, `TocEntry`, `MentatResult`) — defined in `mentat/probes/base.py`
- **Plugin registry** for probes — add new format by implementing `BaseProbe` and registering in `mentat/probes/__init__.py`
- **Two-stage storage** — stubs stored immediately (with ToC); chunks stored after background embedding/summarization
- **Collections** — named doc groups for scoped search; shared storage with doc_id references; LanceDB `WHERE doc_id IN (...)` pre-filtering
- **Config precedence** — explicit arg > `MENTAT_*` env var > default. Separate `api_key`/`api_base` for summary model (`MENTAT_SUMMARY_*`) and embedding model (`MENTAT_EMBEDDING_*`). Global provider keys (`OPENAI_API_KEY`, etc.) read by `litellm` as fallback.
