"""Skill Integration Layer for agent tool use.

Exports OpenAI function calling tool schemas and a system prompt fragment
that teaches agents how to use Mentat as a document intelligence system.

Usage:
    from mentat.skill import get_tool_schemas, get_system_prompt, export_skill

    tools = get_tool_schemas()      # List of OpenAI tool dicts
    prompt = get_system_prompt()    # System prompt fragment string
    skill = export_skill()          # Combined {tools, system_prompt, version}
"""

from typing import Any, Dict, List


SYSTEM_PROMPT = """\
## Document Intelligence System (Mentat)

You have access to Mentat, a document intelligence system that preprocesses \
and indexes all content you interact with — files, web pages, emails, \
documents from Google Drive, and more. Every file you read, every web page \
you fetch, and every external service result (Gmail, Google Drive, etc.) is \
automatically indexed and made available for efficient retrieval.

### How Auto-Indexing Works

Content is indexed automatically when you:
- Read a file (via the Read tool) — indexed with full structure analysis
- Fetch a web page (via WebFetch) — extracted and indexed as HTML
- Receive results from external services (Composio: Gmail, Google Drive, \
etc.) — content extracted and indexed
- Receive file attachments from messaging channels

Once indexed, you can use Mentat's two-step retrieval to access any \
previously-read content efficiently, instead of re-reading entire files.

### Two-Step Retrieval Protocol (Preferred for All File Access)

This is the **default way to read file content**. It saves 80-90% of \
tokens compared to reading entire files with the Read tool.

**Step 1 — Discover**: Use `search_memory` with `toc_only=true` to find \
relevant documents. This returns doc_ids and matched section names — NOT \
full content. Then call `get_doc_meta(doc_id)` to see the document's brief \
intro, table of contents, and reading instructions.

**Step 2 — Read**: Use `read_segment(doc_id, section_path)` to fetch the \
specific section you need. Section names come from get_doc_meta's ToC. \
Parent sections automatically include all child sections' content. You \
only read what you need.

### When to Use Read Tool vs Mentat

| Scenario | Use |
|----------|-----|
| File never read before | `Read` tool (triggers auto-indexing) |
| Re-reading a previously read file | `search_memory` → `read_segment` |
| Finding content across multiple files | `search_memory` (searches all indexed docs) |
| Reading a specific known section | `get_doc_meta` → `read_segment` |
| Quick factual lookup | `search_memory` (standard RAG, no toc_only) |
| Small config/snippet (<50 lines) | `Read` tool is fine |

**Key principle**: when a `<file>` block contains `[Indexed in Mentat — doc_id: ...]` \
instead of full content, the document has been indexed and compressed. You MUST \
use `get_doc_meta(doc_id)` to see its structure, then `read_segment(doc_id, section)` \
to read specific sections. Do NOT ask the user to resend the file. Similarly, if a \
`<mentat-indexed>` tag appears in place of file content, always use `read_segment`.

### Standard RAG Search (Simple Alternative)

For quick, straightforward queries where you just need an answer, use \
`search_memory` without `toc_only`. This returns matching chunks with \
full content and summaries directly — like traditional RAG.

Set `grouped=true` to group results by document (each document appears \
once with all matching chunks nested), which avoids duplicate metadata.

### Source Filtering

All indexed content is tagged by origin. Use the `source` parameter to \
scope searches:
- `openclaw:Read` — files read by agent
- `web_fetch` — web pages fetched
- `composio:gmail` — Gmail content
- `composio:gdrive` — Google Drive documents
- `composio:*` — any Composio service
- `channel:telegram` — files received via Telegram
- `channel:discord` — files received via Discord
- `channel:*` — files from any messaging channel
- `openclaw:memory_store` — explicit memory entries

### Indexing

Use `index_memory` to explicitly add files or raw content. Processing \
happens in the background; use `memory_status` to check progress.

### Guidelines
- **Default to two-step retrieval** for any file >50 lines that was previously read
- Use `Read` tool only for first-time reads or tiny files
- Use `search_memory` to find content across all indexed documents
- Section names from get_doc_meta can be used directly in read_segment
- Use `collection` to scope searches (e.g. "memory" for long-term, "files" for all reads)
- Documents may be processing in the background; check status if results seem incomplete
"""


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search indexed documents. With toc_only=true, returns "
                "doc_ids and matched section names for two-step retrieval "
                "(use with get_doc_meta + read_segment). Without toc_only, "
                "returns full chunk content like standard RAG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                    "toc_only": {
                        "type": "boolean",
                        "description": (
                            "If true, return only doc_ids and matched section "
                            "names (step 1 of two-step protocol). "
                            "If false, return full chunk content (standard RAG)."
                        ),
                        "default": False,
                    },
                    "grouped": {
                        "type": "boolean",
                        "description": (
                            "If true, group results by document — each document "
                            "appears once with all matching chunks nested. "
                            "More token-efficient for multi-document queries."
                        ),
                        "default": False,
                    },
                    "hybrid": {
                        "type": "boolean",
                        "description": (
                            "If true, combine vector similarity with keyword "
                            "matching for better recall on exact terms."
                        ),
                        "default": False,
                    },
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to scope search",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Filter by content source (e.g. 'web_fetch', "
                            "'composio:*', 'composio:gmail'). "
                            "Omit to search all sources."
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_doc_meta",
            "description": (
                "Get a document's metadata: brief intro, table of contents, "
                "instructions, source, and processing status. Use after "
                "search to understand a document's structure and find section "
                "names for read_segment (step 1b of two-step protocol)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from search results",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_segment",
            "description": (
                "Read a specific section from an indexed document (step 2 "
                "of two-step protocol). Use doc_id from search and section "
                "name from get_doc_meta's ToC. Parent sections automatically "
                "include all child sections' content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID from search results",
                    },
                    "section_path": {
                        "type": "string",
                        "description": (
                            "Section name to read (case-insensitive match). "
                            "Use section names from get_doc_meta's ToC."
                        ),
                    },
                    "include_summary": {
                        "type": "boolean",
                        "description": "Include chunk summaries alongside content",
                        "default": True,
                    },
                },
                "required": ["doc_id", "section_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_memory",
            "description": (
                "Index a file or raw text content into the memory system. "
                "Provide either 'path' (file on disk) or both 'content' "
                "and 'filename' (raw text). Returns immediately; processing "
                "happens in the background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to index (alternative to content+filename)",
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Raw text content to index (use with filename, "
                            "alternative to path)"
                        ),
                    },
                    "filename": {
                        "type": "string",
                        "description": (
                            "Logical filename for the content — used for format "
                            "detection and display (required when using content)"
                        ),
                    },
                    "content_type": {
                        "type": "string",
                        "description": (
                            "MIME type hint for raw content (e.g. 'text/html', "
                            "'text/markdown'). Only used with content+filename."
                        ),
                        "default": "text/plain",
                    },
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to organize documents",
                    },
                    "source": {
                        "type": "string",
                        "description": "Origin tag for provenance tracking (e.g. 'upload', 'web_fetch')",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Arbitrary key-value metadata to store with the document",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_status",
            "description": (
                "Check the processing status of an indexed document. "
                "Returns: pending, processing, completed, or failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to check",
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI function calling tool schemas for all memory tools."""
    return TOOL_SCHEMAS


def get_system_prompt() -> str:
    """Return the system prompt fragment teaching the retrieval protocol."""
    return SYSTEM_PROMPT


def export_skill() -> Dict[str, Any]:
    """Export the complete skill definition (tools + system prompt).

    Returns:
        Dict with keys:
            - tools: List of OpenAI function calling tool schemas
            - system_prompt: System prompt fragment for agent instruction
            - version: Skill definition version
            - protocol: Protocol name
    """
    return {
        "tools": TOOL_SCHEMAS,
        "system_prompt": SYSTEM_PROMPT,
        "version": "2.0",
        "protocol": "two-step-retrieval",
    }
