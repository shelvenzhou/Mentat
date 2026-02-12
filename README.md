# Mentat

> **Pure logic. Strategic retrieval.**
>
> _Next-generation Agentic RAG system that transforms "Content Retrieval" into "Strategy Retrieval"._

Mentat solves the **"Token Explosion"** problem in traditional RAG. Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract **semantic fingerprints** — compact representations of structure, metadata, and key content — then a "Librarian" LLM generates **Reading Guides**: actionable instructions on how to efficiently use each file.

---

## Features

- **Semantic Fingerprinting**: Probes extract `Hierarchy + Metadata + Anchors + Snippets` from every file — enough for an LLM to make decisions without reading the full document.
- **Smart Bypass**: Small files (< 1000 tokens) skip skeleton extraction and return full content directly — cheaper than summarizing.
- **13 Format Probes**: PDF, images, Word, PowerPoint, calendars, archives, CSV, JSON, configs, code, logs, Markdown, and HTML — all without LLM calls.
- **Strategy Retrieval**: Returns _instructions_ (e.g., "Filter Column B for values > 100") alongside data.
- **Format-Aware Chunking**: Chunks preserve structural context (section, page, slide, class/function).
- **Collections**: Named groups of documents for scoped search — shared storage, no vector duplication.
- **Hybrid Search**: LanceDB-powered vector + full-text search with reranking.
- **Telemetry**: Built-in tracking of token savings and processing time.

## Architecture

### Layer 1: The Haystack (Physical Storage)

- **LanceDB**: Separate tables for document stubs (metadata + instructions) and chunk-level vectors.
- **FileStore**: Raw file storage with an abstract base for future VirtualFS extensions.
- **ContentHashCache**: SHA-256 deduplication to skip re-indexing identical files.
- **CollectionStore**: Named doc groups as lightweight JSON references.

### Layer 2: The Probes (Semantic Fingerprinting)

Each probe extracts a `ProbeResult` containing: topic summary, hierarchical table of contents (with preview and annotation per entry), statistics, and content chunks. Small files bypass extraction and return full content.

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

### Layer 3: The Librarian (Instruction)

- Uses **litellm** to generate "Reading Guides" from probe results only (never the raw file).
- Supports OpenAI, Claude, Gemini, Ollama, etc.
- Detects `is_full_content` flag to adjust guidance for small files.

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

## Usage

### Python API

```python
import mentat

# Probe a file (no LLM, no storage — just structure extraction)
result = mentat.probe("data/report.pdf")
print(result.topic.title)
print(result.structure.toc)
print(result.chunks)

# Index a file (Probe -> Librarian -> Embed -> Store)
doc_id = await mentat.add("data/report.pdf")

# Search (returns chunks with instructions)
results = await mentat.search("outlier detection", top_k=5)
for r in results:
    print(f"[{r.filename}] {r.section}")
    print(f"  {r.brief_intro}")
    print(f"  Guide: {r.instructions}")

# Inspect an indexed document
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
# Probe a file (rich or JSON output)
mentat probe data/report.pdf --format rich
mentat probe data/dataset.csv --format json

# Index a file (optionally into a collection)
mentat index data/report.pdf
mentat index data/report.pdf -c research_papers

# Search (optionally scoped to a collection)
mentat search "financial summary" --top-k 10 --hybrid
mentat search "financial summary" -c research_papers

# Inspect an indexed document
mentat inspect <doc_id>

# Manage collections
mentat collection list
mentat collection show research_papers
mentat collection delete research_papers
mentat collection remove research_papers <doc_id>

# System stats
mentat stats
```

---

## Telemetry

Mentat automatically tracks performance:

```text
[Stats] Probed: 14.0ms | Librarian: 450ms | Tokens: 150 | Saved: 95.8% context
```

## Extensibility

- **Custom Probes**: Implement `BaseProbe` with `can_handle()` and `run()`, then register in `mentat/probes/__init__.py`.
- **Adaptors**: Implement `BaseAdaptor` to integrate Mentat with external systems (e.g., OpenClaw).
- **VirtualFS** (planned): Extend `BaseFileStore` to provide `ls`, `cat`, `head`, `tail`, `grep`, `find` over stored files.
