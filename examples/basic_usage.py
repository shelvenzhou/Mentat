"""Basic usage: add → search → inspect.

Demonstrates the core workflow of indexing files,
searching for relevant content, and inspecting documents.
"""

import asyncio
import mentat


async def main():
    # Start the background processor (call once on app startup)
    await mentat.start_processor()

    # --- Probe (no LLM, no storage) ---
    # Useful for quick structure extraction without indexing
    result = mentat.probe("README.md")
    print(f"Title: {result.topic.title}")
    print(f"Sections: {len(result.structure.toc)} entries")
    print(f"Chunks: {len(result.chunks)}")

    # --- Index a file ---
    # Default: returns immediately (~1-3s), processes in background
    doc_id = await mentat.add("README.md")
    print(f"\nQueued: {doc_id}")

    # Check processing status
    status = mentat.get_status(doc_id)
    print(f"Status: {status['status']}")  # pending / processing / completed

    # Wait for processing to complete
    await mentat.wait_for(doc_id, timeout=60)
    print("Processing complete!")

    # Or use wait=True to block until done
    doc_id = await mentat.add("pyproject.toml", wait=True)
    print(f"Indexed (sync): {doc_id}")

    # --- Search ---
    results = await mentat.search("installation", top_k=3)
    print(f"\nSearch results ({len(results)}):")
    for r in results:
        print(f"  [{r.filename}] §{r.section} (score={r.score:.3f})")
        if r.summary:
            print(f"    Summary: {r.summary[:100]}...")

    # Grouped by document (no duplicate metadata)
    grouped = await mentat.search_grouped("installation", top_k=3)
    print(f"\nGrouped results ({len(grouped)}):")
    for doc in grouped:
        print(f"  [{doc.filename}] score={doc.score:.3f}")
        for chunk in doc.chunks:
            print(f"    §{chunk.section}")

    # --- Inspect ---
    info = await mentat.inspect(doc_id)
    print(f"\nInspect: {info['filename']}")
    print(f"  Brief intro: {info.get('brief_intro', 'N/A')}")
    print(f"  Chunks: {len(info.get('chunks', []))}")

    # --- Stats ---
    stats = mentat.stats()
    print(f"\nStats: {stats['documents_indexed']} docs indexed")

    # Shutdown (call on app exit)
    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
