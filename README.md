# Mentat

> **Pure logic. Strategic retrieval.**
>
> _Next-generation Agentic RAG system that transforms "Content Retrieval" into "Strategy Retrieval"._

Mentat solves the **"Token Explosion"** problem in traditional RAG. Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** — compact representations of structure, metadata, and key content — then a two-phase Librarian LLM generates per-chunk summaries and **Reading Guides**: actionable instructions on how to efficiently use each file.

---

## Features

- **Semantic Fingerprinting**: Probes extract `Hierarchy + Metadata + Anchors + Snippets` from every file — enough for an LLM to make decisions without reading the full document.
- **Two-Phase Librarian**: Phase 1 summarises each chunk via LLM (batched, concurrent); Phase 2 generates a strategic reading guide from ToC + summaries + statistics — noting what data is present, what's truncated, and how to access the raw file.
- **Smart Bypass**: Small files (< 1000 tokens) skip skeleton extraction and return full content directly — cheaper than summarizing.
- **13 Format Probes**: PDF, images, Word, PowerPoint, calendars, archives, CSV, JSON, configs, code, logs, Markdown, and HTML — all without LLM calls.
- **Strategy Retrieval**: Returns _instructions_ (e.g., "Filter Column B for values > 100") alongside data.
- **Format-Aware Chunking**: Chunks preserve structural context (section, page, slide, class/function). Adjacent small chunks are automatically merged respecting document hierarchy — H1/H2 boundaries are never crossed.
- **Collections**: Named groups of documents for scoped search — shared storage, no vector duplication.
- **Hybrid Search**: LanceDB-powered vector + full-text search with reranking.
- **Multi-Provider LLM**: Powered by `litellm` — works with OpenAI, Claude, Gemini, Ollama, Azure, vLLM, and any OpenAI-compatible endpoint. Separate API keys/base URLs for the summary model and embedding model.
- **Auto Vector Dimensions**: Embedding dimensions are auto-detected from the model name — no manual configuration needed.
- **Telemetry**: Built-in tracking of token savings and processing time across probe, summarise, and librarian phases.
- **⚡ Fast Mode**: Default mode uses template-based instructions and lazy summarization for **19x faster indexing** with near-zero LLM overhead while maintaining semantic fingerprinting benefits.

## Performance

Mentat offers two indexing modes:

| Mode | Indexing Speed | LLM Calls | Use Case |
|------|---------------|-----------|----------|
| **Fast (default)** | ~1.5s/file | Embeddings only | Production indexing, large batches |
| **Full** | ~30s/file | Summaries + Instructions | High-quality summaries needed |

**Benchmark** (single JSON file, 30 chunks):
```
Fast mode: 1.48s  (19.7x faster ⚡)
Full mode: 29.20s
```

**Fast mode** uses template-based instructions and skips LLM summarization during indexing. Summaries can be generated on-demand later. Search quality is unchanged (embeddings use full chunk content).

See [OPTIMIZATIONS.md](OPTIMIZATIONS.md) for detailed performance analysis.

## Architecture

### Layer 1: The Haystack (Physical Storage)

- **LanceDB**: Separate tables for document stubs (metadata + instructions) and chunk-level vectors (with per-chunk summaries).
- **FileStore**: Raw file storage — originals are always kept for downstream detailed access.
- **ContentHashCache**: SHA-256 deduplication to skip re-indexing identical files.
- **CollectionStore**: Named doc groups as lightweight JSON references.

### Layer 2: The Probes (Semantic Fingerprinting)

Each probe extracts a `ProbeResult` containing: topic summary, hierarchical table of contents (with preview and annotation per entry), statistics, and content chunks. Small files bypass extraction and return full content. After structural splitting, adjacent small chunks (< 300 tokens) are merged via `merge_small_chunks()` while respecting H1/H2 hard boundaries, keeping chunks meaningful for embedding and summarisation without crossing major section boundaries.

| Format     | Extension(s)                         | Library          | Extracts                                                             |
| ---------- | ------------------------------------ | ---------------- | -------------------------------------------------------------------- |
| PDF        | `.pdf`                               | `pymupdf`        | Title, ToC (native + font-inferred), captions, per-page chunks       |
| Image      | `.jpg/.png/.gif/.bmp/.tiff/.webp`    | `Pillow`         | Dimensions, format, color mode, EXIF (camera, GPS, date)             |
| Word       | `.docx`                              | `python-docx`    | Heading hierarchy, tables, metadata (author, dates), section chunks  |
| PowerPoint | `.pptx`                              | `python-pptx`    | Slide titles, bullets, speaker notes, image/table counts             |
| Calendar   | `.ics`                               | `icalendar`      | Events with dates, recurrence, attendees, time range                 |
| Archive    | `.zip/.tar/.tar.gz/.tgz/.tar.bz2`   | stdlib           | Directory tree, file type distribution, size stats                   |
| CSV        | `.csv`                               | `pandas`         | Column types, cardinality, null rates, representative rows           |
| JSON       | `.json`                              | stdlib           | Schema tree (depth-limited), per-key chunks, value previews          |
| Config     | `.yaml/.toml/.ini/.conf/.cfg`        | stdlib + `tomli` | Key hierarchy, value types, full content (configs are small)         |
| Code       | `.py/.js/.jsx/.ts/.tsx`              | `tree-sitter`    | Imports, classes, functions, signatures, docstrings, hierarchical ToC|
| Log        | `.log`                               | regex            | Time range, error level stats, format detection, keyword extraction  |
| Markdown   | `.md/.markdown`                      | regex            | Heading hierarchy with preview/annotation, section-aware chunks      |
| Web/HTML   | `.html/.htm`                         | `trafilatura`    | Heading structure, meta tags, semantic elements, section chunks      |

Probes are tried in order (most specific first); first match wins. Image, Word, PowerPoint, and Calendar probes degrade gracefully if their dependencies are missing.

### Layer 3: The Librarian (Two-Phase Instruction Generation)

1. **Phase 1 — Chunk Summarisation**: LLM generates a 1-3 sentence summary for each chunk (batched, concurrent). Small files (`is_full_content`) bypass summarisation — content IS the summary.
2. **Phase 2 — Instruction Generation**: LLM receives ToC + chunk summaries + statistics and produces:
   - `brief_intro`: 1-2 sentence overview
   - `instructions`: actionable reading guide noting what data is present, what is missing/truncated, and how to access the raw file for details

Uses **litellm** — supports OpenAI, Claude, Gemini, Ollama, etc. Original raw files are always stored for future detailed access.

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
# Summary / Librarian model (used for chunk summarisation + instruction generation)
MENTAT_SUMMARY_MODEL=openai/gpt-4o         # any litellm model string
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

# Probe a file (no LLM, no storage — just structure extraction)
result = mentat.probe("data/report.pdf")
print(result.topic.title)
print(result.structure.toc)
print(result.chunks)

# Index a file (fast mode - default, 19x faster ⚡)
doc_id = await mentat.add("data/report.pdf")

# Index with full LLM processing (slower, includes summaries)
doc_id = await mentat.add("data/report.pdf", summarize=True, use_llm_instructions=True)

# Batch indexing with concurrency
import asyncio
files = ["doc1.pdf", "doc2.json", "doc3.md"]
doc_ids = await asyncio.gather(*[mentat.add(f) for f in files])

# Search (returns chunks with summaries and instructions)
results = await mentat.search("outlier detection", top_k=5)
for r in results:
    print(f"[{r.filename}] {r.section}")
    print(f"  Summary: {r.summary}")
    print(f"  Intro: {r.brief_intro}")
    print(f"  Guide: {r.instructions}")

# Inspect an indexed document (includes chunk summaries)
info = await mentat.inspect(doc_id)

# System statistics
mentat.stats()
```

#### Collections

Group files into named collections for scoped search. Documents are indexed once into shared storage — collections hold lightweight references (like soft links), so the same file in multiple collections costs zero extra storage or indexing.

```python
import mentat

# Create / open a collection
papers = mentat.collection("research_papers")

# Add files (indexes if new, just links if already indexed)
await papers.add("paper1.pdf")
await papers.add("paper2.pdf")

# Search within collection only
results = await papers.search("quantum computing")

# Same file in another collection — no re-indexing
physics = mentat.collection("quantum_physics")
await physics.add("paper1.pdf")   # cache hit, links only

# List & manage
papers.list_docs()                # [{doc_id, filename, brief_intro}, ...]
papers.remove(doc_id)             # unlink, not delete
mentat.collections()              # ["research_papers", "quantum_physics"]

# Global search still works across everything
results = await mentat.search("quantum computing")
```

### CLI

```bash
# Probe a file (rich or JSON output) — no LLM needed, runs instantly
mentat probe data/report.pdf --format rich
mentat probe data/dataset.csv --format json

# Index files (fast mode by default - 19x faster ⚡)
mentat index data/report.pdf
mentat index data/*.json -j 5                    # concurrent, 5 files at once
mentat index data/report.pdf -c research_papers  # add to collection

# Full mode (includes LLM summaries + instructions)
mentat index data/report.pdf --summarize --llm-instructions

# Search (optionally scoped to a collection)
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

---

## Testing

```bash
# Run all tests (no API keys needed — LLM calls are mocked)
uv run pytest tests/ -v

# Run specific test files
uv run pytest tests/test_librarian.py -v    # Librarian layer tests
uv run pytest tests/test_config.py -v       # Config + env var tests
uv run pytest tests/test_vector_db.py -v    # LanceDB storage tests
uv run pytest tests/test_cache.py -v        # Content hash cache tests
uv run pytest tests/test_collections.py -v  # Collection store tests
```

## Telemetry

Mentat automatically tracks performance:

```text
[Stats] Probed: 14.0ms | Summarize: 2100.0ms | Librarian: 450.0ms | Tokens: 150 | Saved: 95.8% context
```

## Extensibility

- **Custom Probes**: Implement `BaseProbe` with `can_handle()` and `run()`, then register in `mentat/probes/__init__.py`.
- **Adaptors**: Implement `BaseAdaptor` to integrate Mentat with external systems (e.g., OpenClaw).
- **VirtualFS** (planned): Extend `BaseFileStore` to provide `ls`, `cat`, `head`, `tail`, `grep`, `find` over stored files.
