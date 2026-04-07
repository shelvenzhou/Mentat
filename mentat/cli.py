import asyncio
import json
import logging
import warnings

import click

# Suppress harmless async cleanup warnings on interpreter shutdown.
warnings.filterwarnings("ignore", message="coroutine.*was never awaited")

from mentat.core.hub import Mentat
from mentat import service


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, debug):
    """Mentat: Pure logic. Strategic retrieval."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")
    if ctx.invoked_subcommand is None:
        _print_help(ctx)


def _print_help(ctx):
    """Print a friendly overview of all commands."""
    click.echo()
    click.echo("  Mentat: Pure logic. Strategic retrieval.")
    click.echo()
    click.echo("  Usage: mentat [--debug] <command> [options]")
    click.echo()
    click.echo("  Commands:")
    click.echo("    index <paths>          Index files or directories")
    click.echo("    list                   List all indexed documents")
    click.echo("    search <query>         Search for relevant documents")
    click.echo("    inspect <doc_id>       Show probe results & instructions")
    click.echo("    segment <doc_id> <sec> Read a specific section (step 2)")
    click.echo("    status <doc_id>        Check processing status")
    click.echo("    stats                  Show system statistics")
    click.echo("    probe <files>          Run probes (no LLM, no storage)")
    click.echo("    collection <sub>       Manage collections (list/show/delete/remove)")
    click.echo("    skill                  Export agent tool schemas & prompt")
    click.echo("    serve                  Start HTTP server (port 7832)")
    click.echo("    help                   Show this help message")
    click.echo()
    click.echo("  Run 'mentat <command> --help' for details on a specific command.")
    click.echo()


@cli.command()
@click.pass_context
def help(ctx):
    """Show this help message."""
    _print_help(ctx)


@cli.group("wiki")
def wiki_cmd():
    """Manage the LLM Wiki."""
    pass


@wiki_cmd.command("rebuild")
def wiki_rebuild():
    """Rebuild deterministic wiki pages from indexed documents."""
    m = Mentat.get_instance()
    count = m.wiki_generator.rebuild_all()
    click.echo(f"Rebuilt {count} wiki pages at {m.config.wiki_dir}")


@wiki_cmd.command("url")
def wiki_url():
    """Print the wiki URL (start with `mentat serve` first)."""
    click.echo("http://localhost:7832/wiki/")


def _run_wiki_agent(mode: str, driver: str | None) -> None:
    from mentat.wiki import WikiAgentRunner

    m = Mentat.get_instance()
    runner = WikiAgentRunner(
        wiki_dir=m.config.wiki_dir,
        default_driver=m.config.wiki_agent_driver,
    )
    exit_code = runner.run(mode, driver=driver)
    if exit_code != 0:
        raise SystemExit(exit_code)


@wiki_cmd.command("sync")
@click.option(
    "--driver",
    type=click.Choice(["codex", "claude", "openclaw"]),
    default=None,
    help="Agent driver to use (defaults to MENTAT_WIKI_DRIVER or 'codex').",
)
def wiki_sync(driver):
    """Run an agent sync pass to update topics and index.md."""
    _run_wiki_agent("sync", driver)


@wiki_cmd.command("verify")
@click.option(
    "--driver",
    type=click.Choice(["codex", "claude", "openclaw"]),
    default=None,
    help="Agent driver to use (defaults to MENTAT_WIKI_DRIVER or 'codex').",
)
def wiki_verify(driver):
    """Run an agent verification pass for topic pages."""
    _run_wiki_agent("verify", driver)


@wiki_cmd.command("lint")
@click.option(
    "--driver",
    type=click.Choice(["codex", "claude", "openclaw"]),
    default=None,
    help="Agent driver to use (defaults to MENTAT_WIKI_DRIVER or 'codex').",
)
def wiki_lint(driver):
    """Run an agent lint pass for wiki housekeeping."""
    _run_wiki_agent("lint", driver)


def _resolve_doc_id(m_or_prefix, prefix=None) -> str:
    """Resolve a doc ID prefix to full ID, or exit with a friendly error.

    Accepts either (mentat, prefix) for backward compatibility or just (prefix,).
    """
    if prefix is None:
        # Called as _resolve_doc_id(prefix)
        prefix = m_or_prefix
    try:
        return service.resolve_doc_id(prefix)
    except ValueError as e:
        click.echo(f"Error: {e}")
        raise SystemExit(1)
    except KeyError as e:
        click.echo(str(e))
        raise SystemExit(1)


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
    "--wait",
    is_flag=True,
    help="Wait for background processing to complete (default: async mode)",
)
@click.option(
    "--concurrency",
    "-j",
    type=int,
    default=3,
    help="Number of files to index concurrently (default: 3)",
)
def index(paths, force, coll_name, summarize, llm_instructions, wait, concurrency):
    """Index one or more files or directories.

    By default uses fast template-based instructions and skips LLM summarization.
    This achieves ~10x faster indexing with minimal quality loss.

    NEW: Async mode by default - returns immediately while processing in background.
    Use --wait to block until processing completes (legacy synchronous behavior).

    Examples:
        mentat index file1.json file2.pdf file3.md  # Returns immediately (async)
        mentat index doc.pdf --wait  # Waits for completion (sync)
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
        # Start background processor
        await m.start()

        semaphore = asyncio.Semaphore(concurrency)

        async def index_one(file_path: str, idx: int):
            async with semaphore:
                click.echo(f"  [{idx+1}/{len(file_list)}] {Path(file_path).name}")
                try:
                    if coll_name:
                        coll = m.collection(coll_name)
                        doc_id = await coll.add(
                            file_path,
                            force=force,
                            summarize=summarize,
                            use_llm_instructions=llm_instructions,
                            wait=wait,
                        )
                    else:
                        doc_id = await m.add(
                            file_path,
                            force=force,
                            summarize=summarize,
                            use_llm_instructions=llm_instructions,
                            wait=wait,
                        )

                    # Show status based on wait mode
                    if not wait:
                        status = m.get_processing_status(doc_id)
                        status_indicator = {
                            "pending": "⏳",
                            "processing": "🔄",
                            "completed": "✓",
                            "failed": "❌"
                        }.get(status.get("status", ""), "?")
                        click.echo(f"    {status_indicator} Queued: {doc_id[:8]}…")

                except Exception as e:
                    click.echo(f"    ❌ Error: {e}")

        tasks = [index_one(f, i) for i, f in enumerate(file_list)]
        await asyncio.gather(*tasks)

        # Shutdown processor gracefully
        await m.shutdown()

    asyncio.run(index_batch())


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results to return")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + FTS)")
@click.option(
    "-c", "--collection", "coll_name", default=None, help="Search within a collection"
)
@click.option(
    "--toc-only", is_flag=True, help="Return ToC summaries only (no chunk content)"
)
def search(query, top_k, hybrid, coll_name, toc_only):
    """Search for relevant files/strategies."""
    m = Mentat()
    if coll_name:
        coll = m.collection(coll_name)
        results = asyncio.run(coll.search(query, top_k=top_k, hybrid=hybrid))
    else:
        results = asyncio.run(m.search(query, top_k=top_k, hybrid=hybrid, toc_only=toc_only))
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
@click.option("--full", "-f", is_flag=True, help="Show full ToC and chunk summaries (no truncation)")
def inspect(doc_id, full):
    """Show probe results and instructions for an indexed file."""
    m = Mentat()
    doc_id = _resolve_doc_id(m, doc_id)
    info = asyncio.run(m.inspect(doc_id))
    if not info:
        click.echo(f"Document not found: {doc_id}")
        return

    click.echo(f"\n{'═' * 60}")
    click.echo(f"  Document: {info.get('filename', 'unknown')}")
    click.echo(f"  ID: {doc_id}")
    click.echo(f"{'─' * 60}")
    click.echo(f"  Source: {info.get('source', 'N/A')}")
    click.echo(f"  Brief Intro: {info.get('brief_intro', 'N/A')}")

    # ToC — may be top-level or nested under probe
    toc = info.get("toc", [])
    probe = info.get("probe")
    if not toc and probe:
        structure = probe.get("structure", {})
        toc = structure.get("toc", [])
    if toc:
        click.echo(f"\n{'─' * 60}")
        toc_limit = len(toc) if full else 20
        click.echo(f"  Table of Contents ({len(toc)} entries):")
        for entry in toc[:toc_limit]:
            indent = "    " + "  " * (entry.get("level", 1) - 1)
            page = f" (p.{entry['page']})" if entry.get("page") else ""
            click.echo(f"{indent}- {entry.get('title', '')}{page}")
        if not full and len(toc) > 20:
            click.echo(f"    … and {len(toc) - 20} more")

    # Probe details (topic, stats)
    if probe:
        topic = probe.get("topic", {})
        if topic.get("title"):
            click.echo(f"\n  Title: {topic['title']}")

        stats = probe.get("stats", {})
        if stats:
            click.echo(f"\n{'─' * 60}")
            click.echo(f"  Stats: {json.dumps(stats, indent=6, default=str)}")

    # Chunk summaries
    chunk_summaries = info.get("chunk_summaries")
    if chunk_summaries:
        click.echo(f"\n{'─' * 60}")
        cs_limit = len(chunk_summaries) if full else 15
        click.echo(f"  Chunk Summaries ({len(chunk_summaries)}):")
        for cs in chunk_summaries[:cs_limit]:
            idx = cs.get("index", "?")
            sec = f" [{cs['section']}]" if cs.get("section") else ""
            summary = cs.get("summary", "")[:120]
            click.echo(f"    [{idx}]{sec}: {summary}")


@cli.command()
@click.argument("doc_id")
@click.argument("section")
def segment(doc_id, section):
    """Read a specific section from an indexed document.

    Step 2 of the two-step retrieval protocol.

    Examples:
        mentat segment abc12345 "Installation"
        mentat segment abc12345 "Chapter 1/Setup"
    """
    m = Mentat()
    doc_id = _resolve_doc_id(m, doc_id)
    result = asyncio.run(m.read_segment(doc_id, section))

    if result.get("error"):
        click.echo(f"Error: {result['error']}")
        return

    click.echo(f"\n{'=' * 60}")
    click.echo(f"  Document: {result.get('filename', 'unknown')}")
    click.echo(f"  Section: {result.get('section_path', section)}")
    click.echo(f"  Chunks: {len(result.get('chunks', []))}")
    click.echo(f"{'=' * 60}")

    for chunk in result.get("chunks", []):
        click.echo(f"\n  [{chunk.get('chunk_index', '?')}] {chunk.get('section', '')}")
        preview = chunk.get("content", "")[:500]
        click.echo(f"  {preview}")
        if chunk.get("summary"):
            click.echo(f"  Summary: {chunk['summary'][:200]}")

    if result.get("note"):
        click.echo(f"\n  Note: {result['note']}")


@cli.command()
@click.argument("doc_id")
def status(doc_id):
    """Check processing status for a document.

    Shows whether a document is pending, processing, completed, or failed.
    Useful for tracking async background processing.

    Examples:
        mentat status abc12345
    """
    m = Mentat()
    doc_id = _resolve_doc_id(m, doc_id)
    status_dict = m.get_processing_status(doc_id)

    if status_dict.get("status") == "not_found":
        click.echo(f"❌ Document not found: {doc_id}")
        return

    status_val = status_dict.get("status", "unknown")
    status_icon = {
        "pending": "⏳",
        "processing": "🔄",
        "completed": "✓",
        "failed": "❌"
    }.get(status_val, "?")

    click.echo(f"\n{'═' * 60}")
    click.echo(f"  Document ID: {doc_id}")
    click.echo(f"  Status: {status_icon} {status_val.upper()}")
    click.echo(f"{'─' * 60}")

    if status_dict.get("submitted_at"):
        import time
        elapsed = time.time() - status_dict["submitted_at"]
        click.echo(f"  Submitted: {elapsed:.1f}s ago")

    if status_dict.get("needs_summarization"):
        click.echo(f"  Summarization: Enabled")

    if status_dict.get("error"):
        click.echo(f"  Error: {status_dict['error']}")

    click.echo(f"{'═' * 60}\n")


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
    s = service.get_stats()
    click.echo(f"\n{'═' * 40}")
    click.echo(f"  Mentat System Statistics")
    click.echo(f"{'─' * 40}")
    click.echo(f"  Documents indexed: {s['docs_indexed']}")
    click.echo(f"  Chunks stored:     {s['chunks_stored']}")
    click.echo(f"  Cached hashes:     {s['cached_hashes']}")
    size_mb = s["storage_size_bytes"] / (1024 * 1024)
    click.echo(f"  Raw storage:       {size_mb:.2f} MB")
    click.echo(f"{'═' * 40}")


@cli.command("list")
@click.option(
    "--source",
    "-s",
    default=None,
    help="Filter by source (e.g. read, upload, web_fetch, composio:gmail)",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["table", "json"]),
    default="table",
    help="Output format",
)
def list_docs(source, fmt):
    """List all indexed documents.

    Examples:
        mentat list
        mentat list --source read
        mentat list --format json
    """
    docs = service.list_docs(source=source)

    if not docs:
        click.echo("No documents found.")
        return

    if fmt == "json":
        import json as _json

        out = []
        for d in docs:
            meta = {}
            if d.get("metadata_json"):
                try:
                    meta = _json.loads(d["metadata_json"])
                except Exception:
                    pass
            out.append({
                "id": d.get("id"),
                "filename": d.get("filename"),
                "source": d.get("source"),
                "path": meta.get("path") or meta.get("url"),
            })
        click.echo(_json.dumps(out, indent=2, ensure_ascii=False))
        return

    click.echo(f"\n  {len(docs)} document(s) indexed\n")
    click.echo(f"  {'ID':10s}  {'Source':22s}  Path / Filename")
    click.echo(f"  {'─' * 10}  {'─' * 22}  {'─' * 50}")
    for d in docs:
        doc_id = d.get("id", "?")[:8] + "…"
        src = d.get("source", "?")
        meta = {}
        if d.get("metadata_json"):
            try:
                meta = json.loads(d["metadata_json"])
            except Exception:
                pass
        path = meta.get("path") or meta.get("url") or d.get("filename", "?")
        click.echo(f"  {doc_id:10s}  {src:22s}  {path}")


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
    doc_id = _resolve_doc_id(m, doc_id)
    m.collection(name).remove(doc_id)
    click.echo(f"Removed {doc_id[:8]}… from '{name}'.")


@cli.command()
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["json", "prompt"]),
    default="json",
    help="Output format: json (full export) or prompt (system prompt only)",
)
def skill(fmt):
    """Export agent tool definitions and system prompt.

    Outputs OpenAI function calling tool schemas and a system prompt
    fragment for the two-step retrieval protocol.

    Examples:
        mentat skill                  # Full JSON export
        mentat skill --format prompt  # System prompt only
    """
    from mentat.skill import export_skill, get_system_prompt

    if fmt == "prompt":
        click.echo(get_system_prompt())
    else:
        click.echo(json.dumps(export_skill(), indent=2))


@cli.command()
@click.option("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
@click.option("--port", "-p", default=7832, type=int, help="Port (default: 7832)")
def serve(host, port):
    """Start the Mentat HTTP server.

    Exposes the full Mentat API over HTTP for use by external tools.

    Examples:
        mentat serve
        mentat serve --port 8000
        mentat serve --host 127.0.0.1 --port 9090
    """
    try:
        import uvicorn
    except ImportError:
        click.echo("uvicorn is required: pip install uvicorn[standard]")
        raise SystemExit(1)

    from mentat.server import create_app

    click.echo(f"Starting Mentat server on {host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


if __name__ == "__main__":
    cli()
