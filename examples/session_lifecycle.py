"""Session lifecycle: TTL-based collections for ephemeral workspaces.

Collections support opaque metadata — the host application (e.g. OpenClaw)
can store session info, TTL, or any other data. Mentat stores and returns
metadata but never interprets it (except metadata.ttl for GC).

This pattern is useful for chat sessions, temporary workspaces, or
any context that should auto-expire.
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # ================================================================
    # Create session-scoped collections with TTL
    # ================================================================
    session_id = "ses_abc123"

    # Session collection — expires after 1 hour (3600 seconds)
    mentat.create_collection(
        session_id,
        metadata={
            "ttl": 3600,               # GC will clean up after this
            "user_id": "user_42",       # opaque — mentat doesn't care
            "channel": "slack",         # opaque — mentat doesn't care
        },
        auto_add_sources=[f"session:{session_id}"],
    )

    # ================================================================
    # Use the session — add files, search within scope
    # ================================================================
    session = mentat.collection(session_id)

    # Files added during the session
    id1 = await session.add("uploaded_doc.pdf")
    id2 = await mentat.add_content(
        "User asked about deployment steps...",
        filename="chat_context.md",
        source=f"session:{session_id}",  # auto-routed to session collection
    )

    # Scoped search — only within this session's documents
    results = await session.search("deployment")
    print(f"Session search: {len(results)} results")

    # Check session info
    info = mentat.get_collection_info(session_id)
    print(f"Session {session_id}:")
    print(f"  Docs: {len(info['doc_ids'])}")
    print(f"  Created: {info['created_at']}")
    print(f"  TTL: {info['metadata']['ttl']}s")

    # ================================================================
    # Garbage collection — run periodically to clean expired sessions
    # ================================================================
    # In production, run this on a timer (e.g. every 5 minutes)
    deleted = mentat.gc_collections()
    print(f"\nGC cleaned: {deleted}")
    # After TTL expires, this would return ["ses_abc123"]

    # Documents themselves remain in storage (they might be in other
    # collections too). Only the collection reference is removed.

    # ================================================================
    # Multiple concurrent sessions
    # ================================================================
    for i in range(3):
        sid = f"ses_demo_{i}"
        mentat.create_collection(
            sid,
            metadata={"ttl": 60, "user_id": f"user_{i}"},
        )
    print(f"\nActive collections: {mentat.collections()}")

    # Periodic GC keeps things clean
    # deleted = mentat.gc_collections()

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
