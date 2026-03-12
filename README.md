# Mentat

> **Pure logic. Strategic retrieval.**
>
> _Next-generation Agentic RAG system that transforms "Content Retrieval" into "Strategy Retrieval"._

Mentat solves the **"Token Explosion"** problem in traditional RAG. Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** — compact representations of structure, metadata, and key content — then generates per-chunk summaries (optional) and template-based **Reading Guides**: actionable instructions on how to efficiently use each file.

---

## Features

- **Semantic Fingerprinting**: Probes extract `Hierarchy + Metadata + Anchors + Snippets` from every file — enough for an LLM to make decisions without reading the full document.
- **Smart Instruction Generation**: Template-based reading guides generated from ToC + statistics — noting what data is present, what's truncated, and how to access the raw file. Optional LLM-based chunk summarization available for deeper semantic understanding.
- **Smart Bypass**: Small files (< 1000 tokens) skip skeleton extraction and return full content directly — cheaper than summarizing.
- **13 Format Probes**: PDF, images, Word, PowerPoint, calendars, archives, CSV, JSON, configs, code, logs, Markdown, and HTML — all without LLM calls.
- **Strategy Retrieval**: Returns _instructions_ (e.g., "Filter Column B for values > 100") alongside data.
- **Format-Aware Chunking**: Chunks preserve structural context (section, page, slide, class/function). Adjacent small chunks are automatically merged respecting document hierarchy — H1/H2 boundaries are never crossed.
- **Collections**: Named groups of documents for scoped search — shared storage, no vector duplication. Supports multi-collection search (OR semantics), opaque metadata, and TTL-based garbage collection.
- **Auto-Routing**: Documents are automatically classified into collections based on `source` tags and glob patterns (e.g., `openclaw:*` matches `openclaw:Read`, `openclaw:WebFetch`).
- **File Watcher**: Per-collection directory watching via `watchfiles` (Rust backend). Content-hash dedup and throttling ensure only actual changes trigger re-indexing.
- **Two-Step Retrieval**: Token-efficient search protocol — `search(toc_only=True)` returns lightweight document summaries, then `read_segment()` drills into specific sections on demand.
- **Hybrid Search**: LanceDB-powered vector + full-text search with reranking.
- **Multi-Provider LLM**: Powered by `litellm` — works with OpenAI, Claude, Gemini, Ollama, Azure, vLLM, and any OpenAI-compatible endpoint. Separate API keys/base URLs for the summary model and embedding model.
- **Auto Vector Dimensions**: Embedding dimensions are auto-detected from the model name — no manual configuration needed.
- **Telemetry**: Built-in tracking of token savings and processing time across probe, summarise, and librarian phases.
- **⚡ Fast Mode**: Default mode uses template-based instructions and lazy summarization for **19x faster indexing** with near-zero LLM overhead while maintaining semantic fingerprinting benefits.
- **🚀 Async Processing**: Returns immediately (~1-3s) after probe + ToC extraction while embeddings/summarization process in background. Priority boosting automatically processes queried documents first. Perfect for batch indexing and responsive UIs.
- **🔥 Section Heat Tracking**: Automatically tracks which document sections are most accessed across search, inspect, and read_segment operations. Weighted scoring (read_segment > inspect > search) with exponential time decay, parent-to-child propagation, and a query API for hottest sections.

## Performance

Mentat offers two indexing modes with async/sync processing:

| Mode                       | Async Return Time | Full Processing      | LLM Calls                | Use Case                                           |
| -------------------------- | ----------------- | -------------------- | ------------------------ | -------------------------------------------------- |
| **Fast + Async (default)** | ~1-3s             | ~10-15s (background) | Embeddings only          | Production indexing, responsive UIs, large batches |
| **Fast + Sync**            | ~10-15s           | ~10-15s (blocking)   | Embeddings only          | Legacy compatibility                               |
| **Full + Async**           | ~1-3s             | ~30-60s (background) | Summaries + Instructions | High-quality with responsiveness                   |
| **Full + Sync**            | ~30-60s           | ~30-60s (blocking)   | Summaries + Instructions | High-quality, wait for completion                  |

**Async Mode (NEW)**: Returns immediately after probe + ToC extraction (~1-3s), then processes embeddings/summarization in background. ToC is available immediately for inspection; full vector search available after background processing completes.

**Benchmark** (single JSON file, 30 chunks):

```
Fast + Async:  1.48s → returns immediately (background: +10s)
Fast + Sync:   11.2s (blocking)
Full + Async:  2.1s  → returns immediately (background: +30s)
Full + Sync:   32.5s (blocking)
```

**Fast mode** uses template-based instructions and skips LLM summarization during indexing. Summaries can be generated on-demand later. Search quality is unchanged (embeddings use full chunk content).

**Priority Boosting**: Documents queried before processing completes automatically get higher priority in the background queue (+10 priority boost).

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed performance analysis.

## Architecture

### Layer 1: The Haystack (Physical Storage)

- **LanceDB**: Separate tables for document stubs (metadata + instructions) and chunk-level vectors (with per-chunk summaries).
- **FileStore**: Raw file storage — originals are always kept for downstream detailed access.
- **ContentHashCache**: SHA-256 deduplication to skip re-indexing identical files.
- **CollectionStore**: Named doc groups as lightweight JSON references with opaque metadata, watch configs, auto-routing rules, and TTL-based GC.

### Layer 2: The Probes (Semantic Fingerprinting)

Each probe extracts a `ProbeResult` containing: topic summary, hierarchical table of contents (with preview and annotation per entry), statistics, and content chunks. Small files bypass extraction and return full content. After structural splitting, adjacent small chunks (< 300 tokens) are merged via `merge_small_chunks()` while respecting H1/H2 hard boundaries, keeping chunks meaningful for embedding and summarisation without crossing major section boundaries.

| Format     | Extension(s)                      | Library          | Extracts                                                              |
| ---------- | --------------------------------- | ---------------- | --------------------------------------------------------------------- |
| PDF        | `.pdf`                            | `pymupdf`        | Title, ToC (native + font-inferred), captions, per-page chunks        |
| Image      | `.jpg/.png/.gif/.bmp/.tiff/.webp` | `Pillow`         | Dimensions, format, color mode, EXIF (camera, GPS, date)              |
| Word       | `.docx`                           | `python-docx`    | Heading hierarchy, tables, metadata (author, dates), section chunks   |
| PowerPoint | `.pptx`                           | `python-pptx`    | Slide titles, bullets, speaker notes, image/table counts              |
| Calendar   | `.ics`                            | `icalendar`      | Events with dates, recurrence, attendees, time range                  |
| Archive    | `.zip/.tar/.tar.gz/.tgz/.tar.bz2` | stdlib           | Directory tree, file type distribution, size stats                    |
| CSV        | `.csv`                            | `pandas`         | Column types, cardinality, null rates, representative rows            |
| JSON       | `.json`                           | stdlib           | Schema tree (depth-limited), per-key chunks, value previews           |
| Config     | `.yaml/.toml/.ini/.conf/.cfg`     | stdlib + `tomli` | Key hierarchy, value types, full content (configs are small)          |
| Code       | `.py/.js/.jsx/.ts/.tsx`           | `tree-sitter`    | Imports, classes, functions, signatures, docstrings, hierarchical ToC |
| Log        | `.log`                            | regex            | Time range, error level stats, format detection, keyword extraction   |
| Markdown   | `.md/.markdown`                   | regex            | Heading hierarchy with preview/annotation, section-aware chunks       |
| Web/HTML   | `.html/.htm`                      | `trafilatura`    | Heading structure, meta tags, semantic elements, section chunks       |

Probes are tried in order (most specific first); first match wins. Image, Word, PowerPoint, and Calendar probes degrade gracefully if their dependencies are missing.

### Layer 3: The Librarian (Summarization & Instruction Generation)

1. **Phase 1 — Chunk Summarisation (Optional)**: LLM generates a 1-3 sentence summary for each chunk (batched, concurrent). Small files (`is_full_content`) bypass summarisation — content IS the summary. Can be disabled for faster indexing.
2. **Phase 2 — Instruction Generation**: Template-based generation from ToC + statistics produces:
   - `brief_intro`: 1-2 sentence overview
   - `instructions`: actionable reading guide noting what data is present, what is missing/truncated, and how to access the raw file for details

Uses **litellm** for chunk summarization (when enabled) — supports OpenAI, Claude, Gemini, Ollama, etc. Original raw files are always stored for future detailed access.

---

## Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/mentat.git
cd mentat
uv sync
```

## Configuration

```bash
# Copy the example and fill in your API keys / model names
cp .env.example .env
```

Key settings in `.env`:

```bash
# Summary model (used for optional chunk summarisation)
MENTAT_SUMMARY_MODEL=openai/gpt-4o-mini    # any litellm model string
# MENTAT_SUMMARY_API_KEY=                   # optional, overrides global key
# MENTAT_SUMMARY_API_BASE=                  # optional, custom endpoint

# Embedding model (vector dimension is auto-detected)
MENTAT_EMBEDDING_MODEL=openai/text-embedding-3-small
# MENTAT_EMBEDDING_API_KEY=                 # optional, overrides global key
# MENTAT_EMBEDDING_API_BASE=                # optional, custom endpoint

# Global provider keys (litellm reads these natively as fallback)
OPENAI_API_KEY=sk-...
```

You can use different providers for summarisation and embedding (e.g., Anthropic for summaries, OpenAI for embeddings), each with their own API key and base URL.

## Usage

### Python API

```python
import mentat
import asyncio

# Start background processor (call once on app startup)
await mentat.start_processor()

# Probe a file (no LLM, no storage — just structure extraction)
result = mentat.probe("data/report.pdf")
print(result.topic.title)
print(result.structure.toc)
print(result.chunks)

# Index a file (async mode - returns immediately ⚡)
doc_id = await mentat.add("data/report.pdf")
print(f"Queued: {doc_id}")  # Returns in ~1-3s

# Check processing status
status = mentat.get_status(doc_id)
print(f"Status: {status['status']}")  # pending/processing/completed/failed

# Wait for processing to complete (optional)
completed = await mentat.wait_for(doc_id, timeout=60)
print(f"Processing completed: {completed}")

# Index with synchronous mode (waits for completion)
doc_id = await mentat.add("data/report.pdf", wait=True)  # Blocks until done

# Index with full LLM processing (async by default)
doc_id = await mentat.add("data/report.pdf", summarize=True, use_llm_instructions=True)

# Batch indexing - all return immediately
files = ["doc1.pdf", "doc2.json", "doc3.md"]
doc_ids = await asyncio.gather(*[mentat.add(f) for f in files])
print(f"Queued {len(doc_ids)} documents")  # Returns in seconds!

# Wait for all to complete
await asyncio.gather(*[mentat.wait_for(d) for d in doc_ids])

# Search (returns chunks with summaries and instructions)
# Automatically boosts priority for any queried documents still processing
results = await mentat.search("outlier detection", top_k=5)
for r in results:
    print(f"[{r.filename}] {r.section}")
    print(f"  Summary: {r.summary}")
    print(f"  Intro: {r.brief_intro}")
    print(f"  Guide: {r.instructions}")

# Search grouped by document (no duplicate metadata)
grouped = await mentat.search_grouped("outlier detection", top_k=5)
for doc in grouped:
    print(f"[{doc.filename}] score={doc.score:.3f}")
    print(f"  Intro: {doc.brief_intro}")
    for chunk in doc.chunks:
        print(f"    §{chunk.section}: {chunk.summary}")

# Get document metadata (brief_intro, instructions, toc, status)
meta = await mentat.get_doc_meta(doc_id)
print(f"  Status: {meta['processing_status']}")
print(f"  Intro: {meta['brief_intro']}")

# Inspect an indexed document (includes chunk summaries)
info = await mentat.inspect(doc_id)

# System statistics
mentat.stats()

# Shutdown processor (call on app exit)
await mentat.shutdown()
```

#### Collections

Group files into named collections for scoped search. Documents are indexed once into shared storage — collections hold lightweight references (like symlinks), so the same file in multiple collections costs zero extra storage or indexing.

```python
import mentat

# Create collections with config
mentat.create_collection("code", metadata={"type": "project"})
mentat.create_collection("docs", auto_add_sources=["web_fetch", "openclaw:*"])

# Add files — via collection param or Collection wrapper
await mentat.add("src/main.py", wait=True, collection="code")
code = mentat.collection("code")
await code.add("src/utils.py")

# Scoped search
results = await code.search("authentication")

# Multi-collection search (OR semantics)
results = await mentat.search("auth", collections=["code", "docs"])

# Same file in another collection — no re-indexing (cache hit)
await mentat.add("src/main.py", collection="docs")

# List & manage
mentat.collections()                    # ["code", "docs"]
mentat.get_collection_info("code")      # full record with doc_ids, metadata
mentat.delete_collection("docs")        # removes collection, not documents
```

#### Auto-Routing

Automatically classify documents into collections based on source tags:

```python
# Set up: "openclaw:*" matches any openclaw tool source
mentat.create_collection("files", auto_add_sources=["openclaw:*"])
mentat.create_collection("memory", auto_add_sources=["openclaw:memory"])

# source="openclaw:Read" → auto-routed to "files"
await mentat.add("report.pdf", source="openclaw:Read")

# source="openclaw:memory" → auto-routed to both "files" AND "memory"
await mentat.add("notes.md", source="openclaw:memory")

# Combine auto-routing with explicit collection
await mentat.add("spec.md", source="openclaw:Read", collection="project_x")
# → in both "files" (auto) and "project_x" (explicit)
```

#### File Watcher

Auto-reindex when files change in watched directories:

```python
mentat.create_collection(
    "project",
    watch_paths=["/home/user/project/src"],
    watch_ignore=["node_modules", "*.lock", "__pycache__"],
)

# Watcher starts automatically with Hub.start()
# Changes are: throttled (5s/file) → hash-checked → re-indexed if content changed
```

#### Two-Step Retrieval

Token-efficient search for LLM agents:

```python
# Step 1: Lightweight ToC search (minimal tokens)
results = await mentat.search("auth", toc_only=True, with_metadata=True)
for r in results:
    print(r.filename, r.brief_intro, r.toc_entries)

# Step 2: Drill into specific sections (on demand)
segment = await mentat.read_segment(doc_id, "Authentication Flow")
print(segment["content"])
```

#### Session Lifecycle (TTL-based GC)

Collections support opaque metadata with TTL for ephemeral workspaces:

```python
# Create a session collection that expires after 1 hour
mentat.create_collection("ses_abc", metadata={"ttl": 3600, "user": "alice"})

# Use it normally...
session = mentat.collection("ses_abc")
await session.add("uploaded.pdf")
results = await session.search("deployment")

# Periodic cleanup removes expired collections
deleted = mentat.gc_collections()  # ["ses_abc"] after TTL expires
```

### CLI

```bash
# Probe a file (rich or JSON output) — no LLM needed, runs instantly
mentat probe data/report.pdf --format rich
mentat probe data/dataset.csv --format json

# Index files (async mode by default - returns immediately ⚡)
mentat index data/report.pdf                     # Queues and returns in ~1-3s
mentat index data/*.json -j 5                    # Concurrent, 5 files at once
mentat index data/report.pdf -c research_papers  # Add to collection

# Index with synchronous mode (waits for completion)
mentat index data/report.pdf --wait              # Blocks until fully processed

# Check processing status
mentat status <doc_id>                           # Shows pending/processing/completed/failed

# Full mode (includes LLM summaries + instructions, async by default)
mentat index data/report.pdf --summarize --llm-instructions
mentat index data/report.pdf --summarize --llm-instructions --wait  # Wait for completion

# Search (optionally scoped to a collection)
# Automatically boosts priority for any queried documents still processing
mentat search "financial summary" --top-k 10 --hybrid
mentat search "financial summary" -c research_papers

# Inspect an indexed document (shows probe results + chunk summaries + instructions)
mentat inspect <doc_id>

# Manage collections
mentat collection list
mentat collection show research_papers
mentat collection delete research_papers
mentat collection remove research_papers <doc_id>

# System stats
mentat stats
```

#### Quick CLI Test

```bash
# 1. Probe a file (no API key required — pure local extraction)
mentat probe README.md
mentat probe pyproject.toml --format json

# 2. Set up .env with your API key, then index:
mentat index README.md

# 3. Search your indexed files:
mentat search "installation instructions"

# 4. Inspect the indexed document (shows summaries + reading guide):
mentat inspect <doc_id_from_step_2>
```

### HTTP API (Server Mode)

Run as a standalone HTTP service:

```bash
# Start server (default port 7832)
mentat serve
# or programmatically
uvicorn mentat.server:create_app --factory --host 0.0.0.0 --port 7832
```

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Index a file (supports `source`, `collection`, `metadata`) |
| `POST` | `/index-content` | Index raw text content |
| `POST` | `/search` | Search with vector/hybrid, `collections` filter, `toc_only` |
| `POST` | `/search-grouped` | Search grouped by document |
| `POST` | `/read-segment` | Read a specific section (two-step retrieval step 2) |
| `GET` | `/doc-meta/{id}` | Get document metadata |
| `GET` | `/inspect/{id}` | Full document inspection |
| `POST` | `/collections/{name}` | Create/configure a collection |
| `PUT` | `/collections/{name}` | Update collection config |
| `GET` | `/collections/{name}` | Get collection info |
| `DELETE` | `/collections/{name}` | Delete a collection |
| `GET` | `/collections` | List all collections |
| `POST` | `/collections/gc` | Garbage-collect expired collections |
| `POST` | `/collections/{name}/add` | Add a file to a collection |
| `POST` | `/collections/{name}/search` | Search within a collection |

---

## Examples

See the [`examples/`](examples/) directory for runnable scripts:

| Example | Description |
|---------|-------------|
| [`basic_usage.py`](examples/basic_usage.py) | Core workflow: probe → add → search → inspect |
| [`collections.py`](examples/collections.py) | Named collections, scoped search, multi-collection queries |
| [`auto_routing.py`](examples/auto_routing.py) | Source-based auto-classification into collections |
| [`file_watcher.py`](examples/file_watcher.py) | Directory watching with auto-reindex on changes |
| [`two_step_retrieval.py`](examples/two_step_retrieval.py) | Token-efficient `toc_only` → `read_segment` protocol |
| [`batch_indexing.py`](examples/batch_indexing.py) | Bulk indexing with `add_batch()` and concurrent `add()` |
| [`session_lifecycle.py`](examples/session_lifecycle.py) | TTL-based ephemeral collections with GC |
| [`content_indexing.py`](examples/content_indexing.py) | Index raw strings (chat messages, API responses) |
| [`hybrid_search.py`](examples/hybrid_search.py) | Vector vs hybrid search, source filters, collection scoping |

---

## Testing

The full test suite runs without any API keys — all LLM calls are mocked.

```bash
# Run all tests
uv run pytest tests/ -v

# Run only a specific group
uv run pytest tests/test_smoke.py -v         # End-to-end smoke tests (mock embedding)
uv run pytest tests/test_queue.py -v         # ProcessingQueue & BackgroundProcessor
uv run pytest tests/test_async_summary.py -v # Async summarisation pipeline
uv run pytest tests/test_queue_perf.py -v    # Queue throughput & latency
```

### Test Coverage

| File                     | What it tests                                                                         |
| ------------------------ | ------------------------------------------------------------------------------------- |
| `test_smoke.py`          | End-to-end: probe → add → search → inspect → collections → stats (mock embedding)     |
| `test_queue.py`          | ProcessingQueue (priority, FIFO, bump, cleanup) + BackgroundProcessor lifecycle       |
| `test_async_summary.py`  | Async summarisation flag propagation, status progression, concurrent docs             |
| `test_queue_perf.py`     | Throughput (100 tasks), submit→start latency, memory boundedness, concurrency scaling |
| `test_librarian.py`      | Librarian chunk summarisation, batching, fallback, template generation                |
| `test_probe_utils.py`    | Token estimation, bypass, preview extraction, truncation, format_size, safe_read_text |
| `test_access_tracker.py` | Two-layer FIFO tracker: promotion, eviction, callbacks, stats                         |
| `test_section_heat.py`   | Section heat tracker: weighted scoring, time decay, propagation, persistence           |
| `test_embeddings.py`     | EmbeddingRegistry, LiteLLMEmbedding batching, truncation, order preservation          |
| `test_telemetry.py`      | Telemetry time_it, token/savings recording, format_stats output                       |
| `test_file_store.py`     | LocalFileStore save, get_path, exists, get_size, total_size                           |
| `test_models.py`         | Pydantic model validation: Chunk, TocEntry, ProbeResult serialisation                 |
| `test_config.py`         | Config loading and env var precedence                                                 |
| `test_vector_db.py`      | LanceDB storage: stubs, chunks, search, hybrid, collections                           |
| `test_cache.py`          | Content hash cache deduplication                                                      |
| `test_collections.py`    | CollectionStore: CRUD, metadata, watch config, auto-routing, persistence, GC          |
| `test_collection_search.py` | Auto-routing on add, multi-collection search, Collection class integration         |
| `test_watcher.py`        | File watcher: SHA-256 dedup, filter, sync, throttle, handle_change                    |
| `test_async_workflow.py` | Full async add → search workflow integration                                          |
| `test_source_metadata.py`| Source + metadata provenance through add → search pipeline                            |
| `test_search_grouped.py` | search_grouped: grouping, toc_only, with_metadata, source filter, scoring             |
| `test_doc_meta.py`       | get_doc_meta: fields, not-found, source/metadata, processing status, instructions     |
| `test_server.py`         | HTTP endpoint tests (health, index, search, search-grouped, doc-meta, inspect, etc.)  |
| `test_server_collections.py` | Collection CRUD endpoints, GC, index/search with collection params              |
| `test_skill.py`          | Skill tool schemas (6 tools), system prompt content, export_skill                     |
| `test_performance.py`    | Probe + indexing performance baselines                                                |

## Telemetry

Mentat automatically tracks performance:

```text
[Stats] Probed: 14.0ms | Summarize: 2100.0ms | Librarian: 450.0ms | Tokens: 150 | Saved: 95.8% context
```

## Extensibility

- **Custom Probes**: Implement `BaseProbe` with `can_handle()` and `run()`, then register in `mentat/probes/__init__.py`.
- **Adaptors**: Implement `BaseAdaptor` to integrate Mentat with external systems (e.g., OpenClaw).
- **VirtualFS** (planned): Extend `BaseFileStore` to provide `ls`, `cat`, `head`, `tail`, `grep`, `find` over stored files.
