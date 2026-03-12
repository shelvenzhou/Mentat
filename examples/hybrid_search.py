"""Hybrid search: combining vector and full-text search with filters.

Mentat supports:
  - Vector search (semantic similarity, default)
  - Hybrid search (vector + BM25 full-text, with reranking)
  - Source filtering (exact or glob pattern)
  - Collection scoping (single or multi-collection)
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # --- Set up some test data ---
    mentat.create_collection("code", auto_add_sources=["file:code:*"])
    mentat.create_collection("docs", auto_add_sources=["file:docs:*"])

    await mentat.add("src/auth.py", source="file:code:auth", wait=True)
    await mentat.add("src/db.py", source="file:code:db", wait=True)
    await mentat.add("docs/api.md", source="file:docs:api", wait=True)
    await mentat.add("docs/auth.md", source="file:docs:auth", wait=True)

    # ================================================================
    # Vector-only search (default)
    # ================================================================
    results = await mentat.search("how does authentication work", top_k=5)
    print("=== Vector search ===")
    for r in results:
        print(f"  [{r.filename}] score={r.score:.3f}")

    # ================================================================
    # Hybrid search (vector + full-text with reranking)
    # ================================================================
    # Better for queries that contain specific keywords
    results = await mentat.search(
        "JWT token validation middleware",
        top_k=5,
        hybrid=True,
    )
    print("\n=== Hybrid search ===")
    for r in results:
        print(f"  [{r.filename}] score={r.score:.3f}")

    # ================================================================
    # Source filtering
    # ================================================================
    # Exact match
    results = await mentat.search(
        "authentication",
        source="file:code:auth",
    )
    print(f"\n=== Source filter (exact): {len(results)} results ===")

    # Glob match — all code files
    results = await mentat.search(
        "authentication",
        source="file:code:*",
    )
    print(f"=== Source filter (glob): {len(results)} results ===")

    # ================================================================
    # Collection-scoped search
    # ================================================================
    # Single collection
    results = await mentat.search(
        "authentication",
        collections=["code"],
    )
    print(f"\n=== Code collection only: {len(results)} results ===")

    # Multiple collections (OR semantics)
    results = await mentat.search(
        "authentication",
        collections=["code", "docs"],
    )
    print(f"=== Code + Docs: {len(results)} results ===")

    # ================================================================
    # Combining filters
    # ================================================================
    results = await mentat.search(
        "authentication",
        hybrid=True,
        collections=["code"],
        top_k=3,
    )
    print(f"\n=== Hybrid + collection filter: {len(results)} results ===")
    for r in results:
        print(f"  [{r.filename}] source={r.source} score={r.score:.3f}")

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
