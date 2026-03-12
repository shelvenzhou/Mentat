"""Auto-routing: automatically classify documents into collections by source.

When you call add() with a `source` tag, collections whose `auto_add_sources`
patterns match will automatically receive the document. Uses fnmatch-style
glob matching, so "openclaw:*" matches "openclaw:Read", "openclaw:WebFetch", etc.
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # --- Set up collections with auto_add_sources ---
    # All files from any openclaw tool go to "files"
    mentat.create_collection(
        "files",
        auto_add_sources=["openclaw:*"],
        metadata={"description": "All files from openclaw tools"},
    )

    # Only memory-related files go to "memory"
    mentat.create_collection(
        "memory",
        auto_add_sources=["openclaw:memory"],
        metadata={"description": "Memory and notes"},
    )

    # Web-fetched content goes to "web"
    mentat.create_collection(
        "web",
        auto_add_sources=["web_fetch", "composio:browser"],
        metadata={"description": "Web content"},
    )

    # --- Add with source tags — auto-routing happens automatically ---

    # source="openclaw:Read" matches "openclaw:*" → routed to "files"
    id1 = await mentat.add("report.pdf", source="openclaw:Read", wait=True)
    print(f"report.pdf → collections: {_doc_collections(id1)}")
    # Output: ["files"]

    # source="openclaw:memory" matches both "openclaw:*" AND "openclaw:memory"
    id2 = await mentat.add("notes.md", source="openclaw:memory", wait=True)
    print(f"notes.md → collections: {_doc_collections(id2)}")
    # Output: ["files", "memory"]

    # source="web_fetch" matches "web_fetch" exactly
    id3 = await mentat.add_content(
        "Some scraped content...",
        filename="page.html",
        source="web_fetch",
        wait=True,
    )
    print(f"page.html → collections: {_doc_collections(id3)}")
    # Output: ["web"]

    # --- Combine auto-routing with explicit collection ---
    id4 = await mentat.add(
        "spec.md",
        source="openclaw:Read",   # auto-routes to "files"
        collection="project_x",   # also explicitly added to "project_x"
        wait=True,
    )
    print(f"spec.md → collections: {_doc_collections(id4)}")
    # Output: ["files", "project_x"]

    # --- No source = no auto-routing ---
    id5 = await mentat.add("random.txt", wait=True)
    print(f"random.txt → collections: {_doc_collections(id5)}")
    # Output: []  (not in any collection)

    await mentat.shutdown()


def _doc_collections(doc_id: str) -> list:
    """Helper to get all collections for a doc."""
    store = mentat.Mentat.get_instance().collections_store
    return store.doc_collections(doc_id)


if __name__ == "__main__":
    asyncio.run(main())
