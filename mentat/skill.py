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
token-efficient: you only read what you need.

### Other Tools
- `index_memory`: Add new files or content to the memory system
- `memory_status`: Check if a document has finished processing
- `get_summary`: Get the full overview of a known document

### Guidelines
- Always search before reading -- do not guess section names
- Prefer toc_only search to minimize token usage
- Section names from search results can be used directly in read_segment
- Documents may be processing in the background; check status if chunks are empty
- High-frequency sections are auto-summarized for faster future retrieval
- Use the `source` parameter to scope searches (e.g. "composio:gmail", "web_fetch", "browser")
- Each indexed document has a source tag showing where the content originated
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
                "search_memory results."
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
                            "Use the section names returned by search_memory."
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
                "Get the full structured overview of an indexed document "
                "(ToC, brief intro, instructions, processing status)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "Document ID to inspect",
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
                "Index a file into the memory system for future retrieval. "
                "Returns immediately; processing happens in the background."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to index",
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
                "required": ["path"],
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
