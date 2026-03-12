"""Collections: scoped search over named document groups.

Collections are lightweight reference sets — documents are indexed once
into shared storage, and collections hold pointers (like symlinks).
The same file in multiple collections costs zero extra storage.
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # --- Create collections ---
    mentat.create_collection("code", metadata={"type": "project"})
    mentat.create_collection("docs", metadata={"type": "documentation"})

    # --- Add files to collections ---
    # Method 1: via collection parameter on add()
    id1 = await mentat.add("src/main.py", wait=True, collection="code")
    id2 = await mentat.add("src/utils.py", wait=True, collection="code")
    id3 = await mentat.add("README.md", wait=True, collection="docs")

    # Method 2: via Collection wrapper
    code = mentat.collection("code")
    docs = mentat.collection("docs")
    id4 = await code.add("src/config.py")  # automatically scoped to "code"

    # Same file in multiple collections — no re-indexing (cache hit)
    await mentat.add("README.md", collection="code")  # links only

    # --- Scoped search ---
    # Search only within "code" collection
    results = await code.search("configuration")
    print(f"Code results: {len(results)}")
    for r in results:
        print(f"  [{r.filename}] {r.section}")

    # Search only within "docs" collection
    results = await docs.search("installation")
    print(f"Docs results: {len(results)}")

    # --- Multi-collection search ---
    # Search across multiple collections at once (OR semantics)
    results = await mentat.search("configuration", collections=["code", "docs"])
    print(f"Multi-collection results: {len(results)}")

    # --- Global search (no collection filter) ---
    results = await mentat.search("configuration")
    print(f"Global results: {len(results)}")

    # --- List & manage ---
    print(f"\nAll collections: {mentat.collections()}")

    info = mentat.get_collection_info("code")
    print(f"Code collection: {len(info['doc_ids'])} docs")

    # Remove a doc from collection (doesn't delete the document)
    code_store = mentat.collection("code")
    # mentat.Mentat.get_instance().collections_store.remove_doc("code", id1)

    # Delete entire collection (documents remain in storage)
    # mentat.delete_collection("docs")

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
