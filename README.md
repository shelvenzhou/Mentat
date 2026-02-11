# Mentat

> **Pure logic. Strategic retrieval.**
>
> _Next-generation Agentic RAG system that transforms "Content Retrieval" into "Strategy Retrieval"._

Mentat solves the **"Token Explosion"** problem in traditional RAG. Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract structure and metadata, then a "Librarian" LLM generates **Reading Guides** — actionable instructions on how to efficiently use each file.

---

## 🚀 Features

- **Cost-Efficient**: Non-LLM probes extract structure and stats from files without spending tokens.
- **Strategy Retrieval**: Returns _instructions_ (e.g., "Filter Column B for values > 100") alongside data.
- **Format-Aware Chunking**: Chunks preserve structural context (section, page, class/function).
- **Hybrid Search**: LanceDB-powered vector + full-text search with reranking.
- **Telemetry**: Built-in tracking of token savings and processing time.

## 🏗 Architecture

### Layer 1: The Haystack (Physical Storage)

- **LanceDB**: Separate tables for document stubs (metadata + instructions) and chunk-level vectors.
- **FileStore**: Raw file storage with an abstract base for future VirtualFS extensions.

### Layer 2: The Probes (Statistical)

Mature libraries extract structure without LLMs:

| Format   | Library       | Extracts                                                          |
| -------- | ------------- | ----------------------------------------------------------------- |
| PDF      | `pymupdf`     | Title, ToC (native + visual inference), captions, per-page chunks |
| CSV      | `pandas`      | Columns, `describe()`, null rates, outliers (Z>3), string lengths |
| Markdown | regex         | Header hierarchy, section-aware chunks                            |
| JSON     | stdlib        | Recursive schema tree (keys structure)                            |
| Web/HTML | `trafilatura` | Main content, title, date, domain                                 |
| Python   | `tree-sitter` | Class/function definitions, AST-based chunks                      |

### Layer 3: The Librarian (Instruction)

- Uses **litellm** to generate "Reading Guides" from probe results only (never the raw file).
- Supports OpenAI, Claude, Gemini, Ollama, etc.

---

## 📦 Installation

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/yourusername/mentat.git
cd mentat
uv sync
```

## 🛠 Usage

### Python API

```python
import mentat

# Probe a file (no LLM, no storage — just structure extraction)
result = mentat.probe("data/report.pdf")
print(result.topic.title)
print(result.structure.toc)
print(result.chunks)

# Index a file (Probe → Librarian → Embed → Store)
doc_id = await mentat.add("data/report.pdf")

# Search (returns chunks with instructions)
results = await mentat.search("outlier detection", top_k=5)
for r in results:
    print(f"[{r.filename}] §{r.section}")
    print(f"  {r.brief_intro}")
    print(f"  Guide: {r.instructions}")

# Inspect an indexed document
info = await mentat.inspect(doc_id)

# System statistics
mentat.stats()
```

### CLI

```bash
# Probe a file (rich or JSON output)
mentat probe data/report.pdf --format rich
mentat probe data/dataset.csv --format json

# Index a file
mentat index data/report.pdf

# Search
mentat search "financial summary" --top-k 10 --hybrid

# Inspect an indexed document
mentat inspect <doc_id>

# System stats
mentat stats
```

---

## 📊 Telemetry

Mentat automatically tracks performance:

```text
[Stats] Probed: 14.0ms | Librarian: 450ms | Tokens: 150 | Saved: 95.8% context
```

## 🔌 Extensibility

- **Custom Probes**: Implement `BaseProbe` to support new file formats.
- **Adaptors**: Implement `BaseAdaptor` to integrate Mentat with external systems (e.g., OpenClaw).
- **VirtualFS** (planned): Extend `BaseFileStore` to provide `ls`, `cat`, `head`, `tail`, `grep`, `find` over stored files.
