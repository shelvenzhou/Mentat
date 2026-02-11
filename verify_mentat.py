import asyncio
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path.cwd()))

from mentat.core.hub import Mentat
from mentat.librarian.engine import Librarian
from mentat.core.embeddings import BaseEmbedding, EmbeddingRegistry


# Mock Librarian to avoid API calls during test
class MockLibrarian(Librarian):
    async def generate_guide(self, probe_result):
        return (
            f"Mock summary for {probe_result.filename}",
            f"Mock instructions: Handle this {probe_result.file_type} carefully.",
            10,  # tokens
        )


# Mock Embedding to avoid API calls during test
class MockEmbedding(BaseEmbedding):
    async def embed(self, text: str):
        return [0.1] * 1536


async def run_verification():
    print("=== Mentat Verification ===")

    # Initialize Mentat with mocks
    m = Mentat(db_path="./test_mentat_db", storage_dir="./test_mentat_files")
    m.librarian = MockLibrarian()

    # Override embedding registry for test
    class TestRegistry:
        @classmethod
        def get_provider(cls, name, **kwargs):
            return MockEmbedding()

    import mentat.core.hub

    mentat.core.hub.EmbeddingRegistry = TestRegistry
    m.embeddings = MockEmbedding()

    # 1. Test indexing a CSV
    print("\nIndexing CSV...")
    csv_id = await m.add("samples/test.csv")
    print(f"Indexed CSV with ID: {csv_id}")

    # 2. Test indexing a Markdown
    print("\nIndexing Markdown...")
    md_id = await m.add("samples/test.md")
    print(f"Indexed Markdown with ID: {md_id}")

    # 3. Test Search
    print("\nTesting Search...")
    results = await m.search("test")
    print(f"Found {len(results)} results:")
    for res in results:
        print(f" - {res.filename}: {res.brief_intro}")

    # 4. Test Inspection
    print("\nTesting Inspection...")
    info = await m.inspect(csv_id)
    print(f"Inspection for {csv_id} filename: {info.get('filename')}")

    print("\n=== Verification Complete ===")


if __name__ == "__main__":
    asyncio.run(run_verification())
