import abc
import shutil
from pathlib import Path


class BaseFileStore(abc.ABC):
    """Abstract base for file storage backends.

    A future VirtualFS implementation can extend this to provide:
    - ls(doc_id): list files/chunks
    - cat(doc_id): read full content
    - head(doc_id, n): first n lines
    - tail(doc_id, n): last n lines
    - grep(doc_id, pattern): search within
    - find(pattern): find across stored files
    """

    @abc.abstractmethod
    def save(self, source_path: str, doc_id: str) -> str:
        """Store a file and return its storage path."""
        ...

    @abc.abstractmethod
    def get_path(self, doc_id: str) -> Path:
        """Get the storage path for a document."""
        ...

    @abc.abstractmethod
    def exists(self, doc_id: str) -> bool:
        """Check if a document is stored."""
        ...

    @abc.abstractmethod
    def get_size(self, doc_id: str) -> int:
        """Get file size in bytes."""
        ...


class LocalFileStore(BaseFileStore):
    """Local filesystem storage. Copies raw files to a managed directory."""

    def __init__(self, storage_dir: str = "./mentat_files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(self, source_path: str, doc_id: str) -> str:
        suffix = Path(source_path).suffix
        dest_path = self.storage_dir / f"{doc_id}{suffix}"
        shutil.copy2(source_path, dest_path)
        return str(dest_path)

    def get_path(self, doc_id: str) -> Path:
        # Search for any file starting with doc_id
        for p in self.storage_dir.iterdir():
            if p.stem == doc_id:
                return p
        return self.storage_dir / doc_id

    def exists(self, doc_id: str) -> bool:
        return self.get_path(doc_id).exists()

    def get_size(self, doc_id: str) -> int:
        path = self.get_path(doc_id)
        return path.stat().st_size if path.exists() else 0

    def total_size(self) -> int:
        """Total storage size across all files."""
        return sum(f.stat().st_size for f in self.storage_dir.iterdir() if f.is_file())
