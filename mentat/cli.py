import asyncio
import json
import logging

import click

from mentat.core.hub import Mentat


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """Mentat: Pure logic. Strategic retrieval."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(name)s - %(levelname)s - %(message)s")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def index(path):
    """Index a file or directory."""
    m = Mentat()
    asyncio.run(m.add(path))


@cli.command()
@click.argument("query")
@click.option("--top-k", "-k", default=5, help="Number of results to return")
@click.option("--hybrid", is_flag=True, help="Use hybrid search (vector + FTS)")
def search(query, top_k, hybrid):
    """Search for relevant files/strategies."""
    m = Mentat()
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
                        page = f" (p.{entry.page})" if entry.page else ""
                        click.echo(f"{indent}- {entry.title}{page}")

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
    size_mb = s["storage_size_bytes"] / (1024 * 1024)
    click.echo(f"  Raw storage:       {size_mb:.2f} MB")
    click.echo(f"{'═' * 40}")


if __name__ == "__main__":
    cli()
