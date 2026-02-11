# Mentat

> **Pure logic. Strategic retrieval.**
>
> _Next-generation Agentic RAG system that transforms "Content Retrieval" into "Strategy Retrieval"._

Mentat is designed to solve the **"Token Explosion"** problem in traditional RAG. Instead of feeding raw documents to an LLM, Mentat uses statistical probes to extract structure and metadata, then uses a "Librarian" LLM to generate **"Reading Guides"**.

When you search in Mentat, you don't just get text chunks; you get **Actionable Instructions** on how to read and process the data efficiently.

---

## 🚀 Features

- **Cost-Efficient**: Uses "Probes" (non-LLM code) to extract 80% of the value from files without spending tokens.
- **Strategy Retrieval**: Returns _instructions_ (e.g., "Filter Column B for values > 100") alongside data.
- **Hybrid Search**: Combines semantic understanding with structural metadata.
- **Telemetry**: Built-in tracking of token savings and processing time.

## 🏗 Architecture

### Layer 1: The Haystack (Physical)

- **LanceDB**: Vector storage for stubs (Metadata + Instructions).
- **FileStore**: Raw file backup.

### Layer 2: The Probes (Statistical)

Mature libraries extract structure without LLMs:

- **PDF**: `pymupdf` (ToC, Metadata)
- **CSV**: `pandas` (Null rates, Outliers, Schema)
- **Markdown**: Regex (Header hierarchy, Link density)
- **JSON**: Recursive Schema Inference
- **Web**: `trafilatura` (Main content extraction)
- **Code**: `tree-sitter` (Class/Function definitions)

### Layer 3: The Librarian (Instruction)

- Uses **litellm** to generate "Reading Guides" based on Probe results.
- Supports OpenAI, Claude, Gemini, Ollama, etc.

---

## 📦 Installation

Mentat uses `uv` for dependency management.

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone https://github.com/yourusername/mentat.git
cd mentat
uv sync
```

## 🛠 Usage

### 1. Verification / Demo

Run the included verification script to see Mentat in action with simulated (mock) LLM calls:

```bash
uv run python verify_mentat.py
```

### 2. CLI (Coming Soon)

```bash
# Index a file
mentat index ./data/report.pdf

# Search
mentat search "financial summary"
```

### 3. Python API

```python
from mentat.core.hub import Mentat

m = Mentat()

# Add a file (Probes -> Librarian -> Vector DB)
doc_id = await m.add("data/large_dataset.csv")

# Search (Returns MentatResult with instructions)
results = await m.search("outlier detection")
print(results[0].instructions)
```

---

## 📊 Telemetry

Mentat automatically tracks performance:

```text
[Stats] Probed: 14.0ms | Librarian: 450ms | Tokens: 150 | Saved: 95.8% context
```
