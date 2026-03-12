"""Batch indexing: efficiently index many files at once.

Two approaches:
  1. add_batch() — single API call for batched embedding (fastest)
  2. asyncio.gather() with add() — concurrent individual adds

Both leverage async processing for maximum throughput.
"""

import asyncio
import glob
import mentat


async def main():
    await mentat.start_processor()

    # ================================================================
    # Method 1: add_batch() — batched embedding in one API call
    # ================================================================
    files = glob.glob("docs/*.md")
    if files:
        print(f"Batch indexing {len(files)} files...")
        doc_ids = await mentat.add_batch(
            files,
            source="batch:docs",
            metadata={"batch": "initial-load"},
        )
        print(f"Indexed {len(doc_ids)} documents")
        for path, doc_id in zip(files, doc_ids):
            print(f"  {path} → {doc_id}")

    # ================================================================
    # Method 2: Concurrent add() — more control per file
    # ================================================================
    source_files = glob.glob("src/**/*.py", recursive=True)
    if source_files:
        print(f"\nConcurrent indexing {len(source_files)} files...")

        # Fire-and-forget: all return immediately (async mode)
        tasks = [mentat.add(f, source="batch:src") for f in source_files]
        doc_ids = await asyncio.gather(*tasks)
        print(f"Queued {len(doc_ids)} documents")

        # Wait for all to complete
        await asyncio.gather(*[mentat.wait_for(d, timeout=120) for d in doc_ids])
        print("All processing complete!")

        # Check individual statuses
        for doc_id in doc_ids:
            status = mentat.get_status(doc_id)
            print(f"  {doc_id[:8]}… → {status['status']}")

    # ================================================================
    # Method 3: Batch with collection assignment
    # ================================================================
    mentat.create_collection(
        "project",
        auto_add_sources=["batch:*"],  # auto-route all batch sources
        metadata={"type": "project"},
    )

    # Now any add() with source="batch:..." auto-routes to "project"
    more_files = glob.glob("tests/*.py")
    if more_files:
        doc_ids = await mentat.add_batch(
            more_files,
            source="batch:tests",
        )
        # All docs are now in the "project" collection via auto-routing
        info = mentat.get_collection_info("project")
        print(f"\nProject collection: {len(info['doc_ids'])} docs")

    # ================================================================
    # Stats after bulk indexing
    # ================================================================
    stats = mentat.stats()
    print(f"\nTotal indexed: {stats['documents_indexed']} documents")

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
