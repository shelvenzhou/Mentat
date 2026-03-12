"""Two-step retrieval: token-efficient search for LLM agents.

Instead of dumping full document content into the context window:
  Step 1: search(toc_only=True) → get document-level ToC summaries
  Step 2: read_segment(doc_id, section) → drill into specific sections

This dramatically reduces token usage while maintaining retrieval quality.
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # Index some documents
    await mentat.add("docs/architecture.md", wait=True)
    await mentat.add("docs/api_reference.md", wait=True)
    await mentat.add("src/auth.py", wait=True)

    # ================================================================
    # STEP 1: ToC-only search — lightweight, returns document summaries
    # ================================================================
    results = await mentat.search(
        "authentication flow",
        top_k=5,
        toc_only=True,       # returns ToC entries, not full chunks
        with_metadata=True,   # include brief_intro, instructions
    )

    print("=== Step 1: ToC-only results ===")
    for r in results:
        print(f"\n📄 {r.filename} (score={r.score:.3f})")
        print(f"   Intro: {r.brief_intro}")
        print(f"   Guide: {r.instructions[:100]}...")
        if r.toc_entries:
            print("   Sections:")
            for entry in r.toc_entries:
                print(f"     - {entry.get('title', 'untitled')}")

    # An LLM agent can now decide which sections to read,
    # without having loaded any full content yet.

    # ================================================================
    # STEP 2: Read specific sections — targeted, full content
    # ================================================================
    if results:
        doc_id = results[0].doc_id
        print(f"\n=== Step 2: Reading sections from {results[0].filename} ===")

        # Get document metadata first
        meta = await mentat.get_doc_meta(doc_id)
        if meta and meta.get("toc"):
            # Pick a specific section to read
            section_name = meta["toc"][0].get("title", "")
            if section_name:
                segment = await mentat.read_segment(doc_id, section_name)
                print(f"\n§ {section_name}")
                print(f"  Content: {segment.get('content', 'N/A')[:200]}...")
                if segment.get("summary"):
                    print(f"  Summary: {segment['summary']}")

    # ================================================================
    # Alternative: search_grouped for document-centric results
    # ================================================================
    grouped = await mentat.search_grouped(
        "authentication",
        top_k=3,
        toc_only=True,
        with_metadata=True,
    )
    print("\n=== Grouped ToC results ===")
    for doc in grouped:
        print(f"\n📄 {doc.filename} ({len(doc.chunks)} matching sections)")
        print(f"   Intro: {doc.brief_intro}")
        for chunk in doc.chunks:
            print(f"   - §{chunk.section}")

    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
