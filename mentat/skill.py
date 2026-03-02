"""Skill Integration Layer for agent tool use.

Exports OpenAI function calling tool schemas and a system prompt fragment
that teaches agents how to use Mentat as a memory system.

Usage:
    from mentat.skill import get_tool_schemas, get_system_prompt, export_skill

    tools = get_tool_schemas()      # List of OpenAI tool dicts
    prompt = get_system_prompt()    # System prompt fragment string
    skill = export_skill()          # Combined {tools, system_prompt, version}
"""

from typing import Any, Dict, List


SYSTEM_PROMPT = """\
## Memory System (Mentat)

You have access to a structured memory system for storing and retrieving \
information from indexed documents. The preferred approach is the two-step \
retrieval protocol, which is significantly more token-efficient than \
traditional RAG.

### Two-Step Retrieval Protocol (Preferred)

**Step 1 — Discover**: Use `search_memory` with `toc_only=true` to find \
relevant documents. This returns doc_ids and matched section names — NOT \
full content. Then call `get_doc_meta(doc_id)` to see the document's brief \
intro, table of contents, and reading instructions.

**Step 2 — Read**: Use `read_segment(doc_id, section_path)` to fetch the \
specific section you need. Section names come from get_doc_meta's ToC. \
Parent sections automatically include all child sections' content. You \
only read what you need — typically saving 80-90% of tokens vs full RAG.

### Standard RAG Search (Simple Alternative)

For quick, straightforward queries where you just need an answer, use \
`search_memory` without `toc_only`. This returns matching chunks with \
full content and summaries directly — like traditional RAG.

Set `grouped=true` to group results by document (each document appears \
once with all matching chunks nested), which avoids duplicate metadata.

### Indexing

Use `index_memory` to add files or raw content to the memory system. \
Processing happens in the background; use `memory_status` to check progress.

### Guidelines
- Prefer two-step retrieval for complex or multi-document queries
- Use standard RAG search for simple, direct questions
- Section names from get_doc_meta can be used directly in read_segment
- Documents may be processing in the background; check status if results seem incomplete
- Use the `source` parameter to filter searches by origin (e.g. "composio:gmail", "web_fetch")
- Use `collection` to scope searches to specific document groups
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
