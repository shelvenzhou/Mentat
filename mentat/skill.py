"""Skill Integration Layer for agent tool use.

Exports OpenAI function calling tool schemas and a system prompt fragment
that teaches agents the two-step retrieval protocol for using Mentat as
a memory system.

Usage:
    from mentat.skill import get_tool_schemas, get_system_prompt, export_skill

    tools = get_tool_schemas()      # List of OpenAI tool dicts
    prompt = get_system_prompt()    # System prompt fragment string
    skill = export_skill()          # Combined {tools, system_prompt, version}
"""

from typing import Any, Dict, List


SYSTEM_PROMPT = """\
## Memory System (Mentat)

You have access to a structured memory system for retrieving information from \
indexed documents. Use the two-step retrieval protocol:

### Step 1: Discover (search_memory)
Search for relevant documents using `search_memory` with `toc_only=true`. \
This returns document summaries and tables of contents -- NOT full content. \
Use this to understand what information is available and where it lives.

### Step 2: Read (read_segment)
Once you identify a relevant section from Step 1, use `read_segment` with \
the doc_id and section name to retrieve the actual content. This is \
token-efficient: you only read what you need. Parent sections automatically \
include all child sections' content.

### Other Tools
- `index_memory`: Add files (via path) or raw content (via content+filename)
- `memory_status`: Check if a document has finished processing
- `get_summary`: Get a document overview (lightweight by default; use \
`sections` for specific sections or `full=true` for everything)
- `get_doc_meta`: Get document metadata (brief_intro, instructions, ToC, \
source, status) by doc_id. Useful after search with `with_metadata=false` \
to retrieve metadata for a specific document on demand.
- `get_section_heat`: Query which sections are most accessed. Returns \
sections ranked by decayed importance score. Use to understand what \
content is most relevant, or to prioritize re-reading hot sections.

### Guidelines
- Always search before reading -- do not guess section names
- Prefer toc_only search to minimize token usage
- For subsequent searches on known documents, the `with_metadata` flag \
defaults to false to save tokens; set it to true if you need brief_intro \
and instructions again
- Section names from search results can be used directly in read_segment
- Documents may be processing in the background; check status if chunks are empty
- The system tracks section access heat automatically: read_segment (weight 3), \
inspect (weight 2), and search (weight 1) all contribute to importance scores \
with 24-hour exponential decay
- Use `get_section_heat` to discover the hottest sections across documents
- Use the `source` parameter to scope searches (e.g. "composio:gmail", "web_fetch", "browser")
"""


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search indexed documents for relevant content. "
                "Use toc_only=true for the discovery step (returns document "
                "summaries and matched sections, not full content). "
                "Use toc_only=false to get full chunk content."
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
                            "If true, return only document summaries and "
                            "matched section names (step 1 of two-step "
                            "protocol). If false, return full chunk content."
                        ),
                        "default": True,
                    },
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to scope search",
                    },
                    "source": {
                        "type": "string",
                        "description": (
                            "Filter by content source. Exact match or glob "
                            "prefix (e.g. 'web_fetch', 'composio:*', "
                            "'composio:gmail'). Omit to search all sources."
                        ),
                    },
                    "with_metadata": {
                        "type": "boolean",
                        "description": (
                            "Include brief_intro, instructions, and toc_entries "
                            "in results. Defaults to true when toc_only=true "
                            "(discovery needs metadata), false when toc_only=false "
                            "(saves tokens when you already know the documents)."
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
            "name": "read_segment",
            "description": (
                "Read a specific section from an indexed document (step 2 of "
                "two-step protocol). Use doc_id and section name from "
                "search_memory results. Parent sections automatically "
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
                            "Use the section names returned by search_memory. "
                            "Parent sections automatically include child content."
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
            "name": "get_summary",
            "description": (
                "Get the structured overview of an indexed document. "
                "Returns lightweight data (ToC + brief intro) by default. "
                "Use 'sections' to get chunk summaries for specific sections, "
                "or 'full' to get everything including the full probe data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to inspect",
                    },
                    "sections": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of section names to include "
                            "chunk summaries for (case-insensitive match)"
                        ),
                    },
                    "full": {
                        "type": "boolean",
                        "description": (
                            "If true, return all probe data, chunks, and "
                            "summaries. Default false (lightweight ToC + intro)."
                        ),
                        "default": False,
                    },
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "index_memory",
            "description": (
                "Index a file or raw text content into the memory system for "
                "future retrieval. Provide either 'path' (file on disk) or "
                "both 'content' and 'filename' (raw text). "
                "Returns immediately; processing happens in the background."
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
                    "collection": {
                        "type": "string",
                        "description": "Optional collection name to organize documents",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait for processing to complete before returning",
                        "default": False,
                    },
                    "source": {
                        "type": "string",
                        "description": "Origin tag for provenance tracking (e.g. 'upload', 'workspace')",
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
    {
        "type": "function",
        "function": {
            "name": "get_doc_meta",
            "description": (
                "Get lightweight metadata for a document by ID. "
                "Returns brief_intro, instructions, table of contents, "
                "source, metadata, and processing status. "
                "Use after search to get document context without re-fetching."
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
            "name": "get_section_heat",
            "description": (
                "Query which document sections are most accessed. Returns "
                "sections ranked by importance score (with exponential time "
                "decay). Optionally filter by doc_id. Use to discover hot "
                "sections or prioritize re-reading frequently accessed content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": (
                            "Optional document ID to filter results. "
                            "Omit to get hottest sections across all documents."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of sections to return (default: 20)",
                        "default": 20,
                    },
                },
                "required": [],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI function calling tool schemas for all memory tools."""
    return TOOL_SCHEMAS


def get_system_prompt() -> str:
    """Return the system prompt fragment teaching the two-step protocol."""
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
        "version": "1.0",
        "protocol": "two-step-retrieval",
    }
