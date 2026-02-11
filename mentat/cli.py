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
@click.argument("file_path", type=click.Path(exists=True))
def probe(file_path):
    """Debug a file's probe results directly."""
    click.echo(f"Probing {file_path}...")
    # TODO: implement direct probe call


if __name__ == "__main__":
    cli()
