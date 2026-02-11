import asyncio
import click
import logging
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
def search(query):
    """Search for relevant files/strategies."""
    m = Mentat()
    results = asyncio.run(m.search(query))
    for res in results:
        click.echo(f"[{res.filename}] {res.brief_intro}")
        click.echo(f"  Guide: {res.instructions}")


@cli.command()
@click.argument("doc_id")
def inspect(doc_id):
    """Show probes and instructions for an indexed file."""
    m = Mentat()
    info = asyncio.run(m.inspect(doc_id))
    click.echo(info)


@cli.command()
@click.argument("file_paths", nargs=-1, type=click.Path(exists=True))
def probe(file_paths):
    """Run probes on one or more files and print results."""
    from mentat.probes import get_probe

    if not file_paths:
        click.echo("No files provided.")
        return

    for file_path in file_paths:
        click.echo(f"\n--- Probing {file_path} ---")
        probe_instance = get_probe(file_path)

        if not probe_instance:
            click.echo(f"No suitable probe found for {file_path}")
            continue

        try:
            result = probe_instance.run(file_path)
            # Print result in a readable format
            click.echo(f"File Type: {result.file_type}")

            import json

            click.echo("Stats:")
            click.echo(json.dumps(result.stats, indent=2, default=str))

            click.echo("Structure:")
            click.echo(json.dumps(result.structure, indent=2, default=str))

            click.echo(f"Summary Hint: {result.summary_hint}")
            if result.raw_snippet:
                click.echo(f"Snippet: {result.raw_snippet[:100]}...")
        except Exception as e:
            click.echo(f"Error probing {file_path}: {e}")


if __name__ == "__main__":
    cli()
