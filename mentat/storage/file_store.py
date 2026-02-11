import shutil
import os
from pathlib import Path


class FileStore:
    def __init__(self, storage_dir: str = "./mentat_files"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, source_path: str, doc_id: str) -> str:
        dest_path = self.storage_dir / f"{doc_id}_{Path(source_path).name}"
        shutil.copy2(source_path, dest_path)
        return str(dest_path)

    def get_path(self, doc_id: str, original_filename: str) -> Path:
        return self.storage_dir / f"{doc_id}_{original_filename}"
