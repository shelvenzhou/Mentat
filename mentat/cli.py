import asyncio
import json
import logging
import warnings

import click

# litellm creates fire-and-forget async logging coroutines that get
# cancelled when asyncio.run() tears down the event loop.  Harmless.
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

from mentat.core.hub import Mentat


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """Mentat: Pure logic. Strategic retrieval."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True), required=True)
@click.option(
    "--force", is_flag=True, help="Re-index even if file was already processed"
)
@click.option(
    "-c", "--collection", "coll_name", default=None, help="Add to a named collection"
)
@click.option(
    "--summarize",
    is_flag=True,
    help="Enable LLM-based chunk summarization (slow, high quality)",
)
@click.option(
    "--llm-instructions",
    is_flag=True,
    help="Use LLM for instruction generation (slow, smart)",
)
@click.option(
    "--concurrency",
    "-j",
    type=int,
    default=3,
    help="Number of files to index concurrently (default: 3)",
)
def index(paths, force, coll_name, summarize, llm_instructions, concurrency):
    """Index one or more files or directories.

    By default uses fast template-based instructions and skips LLM summarization.
    This achieves ~10x faster indexing with minimal quality loss.

    Examples:
        mentat index file1.json file2.pdf file3.md
        mentat index samples/files/*.json --concurrency 5
        mentat index doc.pdf --summarize --llm-instructions  # Full quality, slow
    """
    import os
    from pathlib import Path

    m = Mentat()

    # Expand paths (handle directories and globs)
    file_list = []
    for path_str in paths:
        p = Path(path_str)
        if p.is_dir():
            # Recursively find files
            file_list.extend([str(f) for f in p.rglob("*") if f.is_file()])
        else:
            file_list.append(str(p))

    if not file_list:
        click.echo("No files to index.")
        return

    click.echo(f"Indexing {len(file_list)} file(s)...")
    if concurrency > 1:
        click.echo(f"Concurrency: {concurrency}")
    click.echo(f"{'─' * 60}\n")

    async def index_batch():
        """Index files with controlled concurrency."""
        semaphore = asyncio.Semaphore(concurrency)

        async def index_one(file_path: str, idx: int):
            async with semaphore:
                click.echo(f"  [{idx+1}/{len(file_list)}] {Path(file_path).name}")
                try:
                    if coll_name:
                        coll = m.collection(coll_name)
                        await coll.add(
                            file_path,
                            force=force,
                            summarize=summarize,
                            use_llm_instructions=llm_instructions,
                        )
                    else:
                        await m.add(
                            file_path,
                            force=force,
                            summarize=summarize,
                            use_llm_instructions=llm_instructions,
                        )
                except Exception as e:
                    click.echo(f"    ❌ Error: {e}")

        tasks = [index_one(f, i) for i, f in enumerate(file_list)]
        await asyncio.gather(*tasks)

    asyncio.run(index_batch())


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results to return")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + FTS)")
@click.option(
    "-c", "--collection", "coll_name", default=None, help="Search within a collection"
)
def search(query, top_k, hybrid, coll_name):
    """Search for relevant files/strategies."""
    m = Mentat()
    if coll_name:
        coll = m.collection(coll_name)
        results = asyncio.run(coll.search(query, top_k=top_k, hybrid=hybrid))
    else:
        results = asyncio.run(m.search(query, top_k=top_k, hybrid=hybrid))
    if not results:
        click.echo("No results found.")
        return

    for i, res in enumerate(results, 1):
        click.echo(f"\n{'─' * 60}")
        click.echo(
            f"  [{i}] {res.filename}" + (f"  §{res.section}" if res.section else "")
        )
        click.echo(f"  Score: {res.score:.4f}")
        click.echo(f"  Intro: {res.brief_intro}")
        click.echo(f"  Guide: {res.instructions}")
        if res.summary:
            click.echo(f"  Summary: {res.summary}")
        if res.content:
            preview = res.content[:200].replace("\n", " ")
            click.echo(f"  Content: {preview}...")


@cli.command()
@click.argument("doc_id")
def inspect(doc_id):
    """Show probe results and instructions for an indexed file."""
    m = Mentat()
    info = asyncio.run(m.inspect(doc_id))
    if not info:
        click.echo(f"Document not found: {doc_id}")
        return

    click.echo(f"\n{'═' * 60}")
    click.echo(f"  Document: {info.get('filename', 'unknown')}")
    click.echo(f"  ID: {doc_id}")
    click.echo(f"{'─' * 60}")
    click.echo(f"  Brief Intro: {info.get('brief_intro', 'N/A')}")
    click.echo(f"  Instructions: {info.get('instruction', 'N/A')}")

    probe = info.get("probe")
    if probe:
        click.echo(f"\n{'─' * 60}")
        click.echo("  Probe Results:")

        topic = probe.get("topic", {})
        if topic.get("title"):
            click.echo(f"    Title: {topic['title']}")

        structure = probe.get("structure", {})
        toc = structure.get("toc", [])
        if toc:
            click.echo(f"    ToC ({len(toc)} entries):")
            for entry in toc[:10]:
                indent = "      " + "  " * (entry.get("level", 1) - 1)
                click.echo(f"{indent}- {entry.get('title', '')}")

        stats = probe.get("stats", {})
        if stats:
            click.echo(f"    Stats: {json.dumps(stats, indent=6, default=str)}")

    # Chunk summaries
    chunk_summaries = info.get("chunk_summaries")
    if chunk_summaries:
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  Chunk Summaries ({len(chunk_summaries)}):")
        for cs in chunk_summaries[:15]:
            idx = cs.get("index", "?")
            sec = f" [{cs['section']}]" if cs.get("section") else ""
            summary = cs.get("summary", "")[:120]
            click.echo(f"    [{idx}]{sec}: {summary}")


@cli.command()
@click.argument("file_paths", nargs=-1, type=click.Path(exists=True))
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["rich", "json"]),
    default="rich",
    help="Output format",
)
def probe(file_paths, fmt):
    """Run probes on one or more files and print results (no LLM, no storage)."""
    from mentat.probes import run_probe

    if not file_paths:
        click.echo("No files provided.")
        return

    for file_path in file_paths:
        click.echo(f"\n{'═' * 60}")
        click.echo(f"  Probing: {file_path}")
        click.echo(f"{'═' * 60}")

        try:
            result = run_probe(file_path)

            if fmt == "json":
                click.echo(
                    json.dumps(
                        result.model_dump(), indent=2, default=str, ensure_ascii=False
                    )
                )
            else:
                # Rich format
                click.echo(f"  File Type: {result.file_type}")

                # Topic
                if result.topic.title:
                    click.echo(f"  Title: {result.topic.title}")
                if result.topic.abstract:
                    click.echo(f"  Abstract: {result.topic.abstract[:200]}...")
                if result.topic.first_paragraph:
                    click.echo(f"  First Para: {result.topic.first_paragraph[:200]}...")

                # Structure
                if result.structure.toc:
                    click.echo(
                        f"\n  Table of Contents ({len(result.structure.toc)} entries):"
                    )
                    for entry in result.structure.toc[:15]:
                        indent = "    " + "  " * (entry.level - 1)
                        annot = f" ({entry.annotation})" if entry.annotation else ""
                        page = f" (p.{entry.page})" if entry.page else ""
                        preview = f" — {entry.preview}" if entry.preview else ""
                        click.echo(f"{indent}- {entry.title}{annot}{page}{preview}")

                if result.structure.captions:
                    click.echo(f"\n  Captions ({len(result.structure.captions)}):")
                    for cap in result.structure.captions[:10]:
                        click.echo(f"    - [{cap.kind}] {cap.text[:80]}")

                if result.structure.columns:
                    click.echo(f"\n  Columns: {', '.join(result.structure.columns)}")

                if result.structure.definitions:
                    click.echo(
                        f"\n  Definitions ({len(result.structure.definitions)}):"
                    )
                    for d in result.structure.definitions[:20]:
                        click.echo(f"    - {d}")

                if result.structure.schema_tree:
                    click.echo(
                        f"\n  Schema Tree: {json.dumps(result.structure.schema_tree, default=str)[:300]}"
                    )

                # Stats
                if result.stats:
                    click.echo(f"\n  Stats:")
                    for k, v in result.stats.items():
                        if isinstance(v, dict):
                            click.echo(f"    {k}: <{len(v)} entries>")
                        else:
                            click.echo(f"    {k}: {v}")

                # Chunks
                click.echo(f"\n  Chunks: {len(result.chunks)} total")
                if result.chunks:
                    for chunk in result.chunks[:3]:
                        section = f" [{chunk.section}]" if chunk.section else ""
                        preview = chunk.content[:100].replace("\n", " ")
                        click.echo(f"    [{chunk.index}]{section}: {preview}...")

        except Exception as e:
            click.echo(f"  Error: {e}")


@cli.command()
def stats():
    """Show system statistics."""
    m = Mentat()
    s = m.stats()
    click.echo(f"\n{'═' * 40}")
    click.echo(f"  Mentat System Statistics")
    click.echo(f"{'─' * 40}")
    click.echo(f"  Documents indexed: {s['docs_indexed']}")
    click.echo(f"  Chunks stored:     {s['chunks_stored']}")
    click.echo(f"  Cached hashes:     {s['cached_hashes']}")
    size_mb = s["storage_size_bytes"] / (1024 * 1024)
    click.echo(f"  Raw storage:       {size_mb:.2f} MB")
    click.echo(f"{'═' * 40}")


@cli.group("collection")
def collection_cmd():
    """Manage collections (named groups of documents)."""
    pass


@collection_cmd.command("list")
def collection_list():
    """List all collections."""
    m = Mentat()
    names = m.list_collections()
    if not names:
        click.echo("No collections.")
        return
    for name in names:
        coll = m.collection(name)
        click.echo(f"  {name}  ({len(coll.doc_ids)} docs)")


@collection_cmd.command("show")
@click.argument("name")
def collection_show(name):
    """Show documents in a collection."""
    m = Mentat()
    coll = m.collection(name)
    docs = coll.list_docs()
    if not docs:
        click.echo(f"Collection '{name}' is empty or does not exist.")
        return
    click.echo(f"\n  Collection: {name} ({len(docs)} docs)")
    click.echo(f"{'─' * 60}")
    for doc in docs:
        click.echo(f"  {doc['doc_id'][:8]}…  {doc['filename']}")
        if doc.get("brief_intro"):
            click.echo(f"    {doc['brief_intro'][:80]}")


@collection_cmd.command("delete")
@click.argument("name")
def collection_delete(name):
    """Delete a collection (does NOT delete the indexed documents)."""
    m = Mentat()
    if m.collection(name).delete():
        click.echo(f"Deleted collection '{name}'.")
    else:
        click.echo(f"Collection '{name}' not found.")


@collection_cmd.command("remove")
@click.argument("name")
@click.argument("doc_id")
def collection_remove(name, doc_id):
    """Remove a document from a collection (does NOT delete from storage)."""
    m = Mentat()
    m.collection(name).remove(doc_id)
    click.echo(f"Removed {doc_id[:8]}… from '{name}'.")


if __name__ == "__main__":
    cli()
