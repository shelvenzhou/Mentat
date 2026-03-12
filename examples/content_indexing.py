"""Content indexing: index raw strings without files on disk.

Useful for indexing chat messages, API responses, web scrapes,
clipboard content, or any text that doesn't originate from a file.
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # ================================================================
    # Index raw text content
    # ================================================================

    # Chat message
    id1 = await mentat.add_content(
        content="The deployment uses Kubernetes with 3 replicas behind an nginx ingress.",
        filename="chat_2024-03-12.md",
        source="chat:slack",
        metadata={"channel": "#ops", "author": "alice"},
        wait=True,
    )
    print(f"Chat message: {id1}")

    # API response
    id2 = await mentat.add_content(
        content='{"status": "healthy", "version": "2.1.0", "uptime": "14d 3h"}',
        filename="health_check.json",
        content_type="application/json",
        source="api:monitoring",
        wait=True,
    )
    print(f"API response: {id2}")

    # Web scrape
    id3 = await mentat.add_content(
        content="""
        # Kubernetes Best Practices
        ## Resource Limits
        Always set CPU and memory limits for your pods.
        ## Health Checks
        Use liveness and readiness probes for all services.
        """,
        filename="k8s_best_practices.md",
        source="web_fetch",
        collection="knowledge_base",
        wait=True,
    )
    print(f"Web content: {id3}")

    # ================================================================
    # Search across all content types
    # ================================================================
    results = await mentat.search("kubernetes deployment", top_k=5)
    print(f"\nSearch results ({len(results)}):")
    for r in results:
        print(f"  [{r.filename}] {r.source} — score={r.score:.3f}")
        print(f"    {r.content[:100]}...")

    # Filter by source
    results = await mentat.search("deployment", source="chat:*")
    print(f"\nChat-only results: {len(results)}")

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
