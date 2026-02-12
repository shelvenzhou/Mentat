import os
import zipfile
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes._utils import format_size


class ArchiveProbe(BaseProbe):
    """Probe for archive files (ZIP, TAR)."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith(
            (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
        )

    def run(self, file_path: str) -> ProbeResult:
        lower = file_path.lower()

        if lower.endswith(".zip"):
            entries, archive_format = self._list_zip(file_path), "zip"
        else:
            entries, archive_format = self._list_tar(file_path), "tar"

        # entries: list of (path, size, is_dir)
        file_entries = [(p, s) for p, s, d in entries if not d]
        dir_entries = set()
        for p, _, _ in entries:
            parts = Path(p).parts
            for i in range(1, len(parts)):
                dir_entries.add("/".join(parts[:i]))

        total_size = sum(s for _, s in file_entries)
        file_count = len(file_entries)
        dir_count = len(dir_entries)

        # --- File type distribution ---
        ext_counter: Counter = Counter()
        for p, _ in file_entries:
            ext = Path(p).suffix.lower()
            if ext:
                ext_counter[ext] += 1
            else:
                ext_counter["(no ext)"] += 1
        file_type_distribution = dict(ext_counter.most_common(15))

        # --- Directory tree as ToC (max depth 3) ---
        toc_entries = self._build_dir_toc(entries, max_depth=3)

        # --- Stats ---
        stats: Dict[str, Any] = {
            "file_count": file_count,
            "dir_count": dir_count,
            "total_uncompressed_size": total_size,
            "total_size_human": format_size(total_size),
            "archive_format": archive_format,
            "file_type_distribution": file_type_distribution,
        }

        # --- Topic ---
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=(
                f"{archive_format.upper()} archive with {file_count} files "
                f"({format_size(total_size)}) in {dir_count} directories"
            ),
        )

        structure = StructureInfo(toc=toc_entries)

        # --- Single chunk: file listing ---
        listing_lines = []
        for p, s, d in entries[:100]:
            if not d:
                listing_lines.append(f"{p}  ({format_size(s)})")
        listing = "\n".join(listing_lines)
        if len(entries) > 100:
            listing += f"\n... and {len(entries) - 100} more entries"

        chunks = [Chunk(content=listing, index=0, section="file_listing")]

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="archive",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=listing[:500],
        )

    def _list_zip(
        self, file_path: str
    ) -> List[Tuple[str, int, bool]]:
        result = []
        with zipfile.ZipFile(file_path) as zf:
            for info in zf.infolist():
                is_dir = info.filename.endswith("/")
                result.append((info.filename, info.file_size, is_dir))
        return result

    def _list_tar(
        self, file_path: str
    ) -> List[Tuple[str, int, bool]]:
        result = []
        with tarfile.open(file_path) as tf:
            for member in tf.getmembers():
                result.append((member.name, member.size, member.isdir()))
        return result

    def _build_dir_toc(
        self,
        entries: List[Tuple[str, int, bool]],
        max_depth: int = 3,
    ) -> List[TocEntry]:
        """Build directory tree ToC with file counts and sizes per directory."""
        # Aggregate stats per directory
        dir_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"files": 0, "size": 0}
        )

        for path, size, is_dir in entries:
            if is_dir:
                continue
            parts = Path(path).parts
            # Attribute file to each ancestor directory
            for depth in range(1, min(len(parts), max_depth + 1)):
                dir_path = "/".join(parts[:depth])
                dir_stats[dir_path]["files"] += 1
                dir_stats[dir_path]["size"] += size

        # Build ToC from sorted unique directories
        toc: List[TocEntry] = []
        seen = set()
        for path, _, _ in entries:
            parts = Path(path).parts
            for depth in range(1, min(len(parts), max_depth + 1)):
                dir_path = "/".join(parts[:depth])
                if dir_path not in seen:
                    seen.add(dir_path)
                    ds = dir_stats.get(dir_path, {"files": 0, "size": 0})
                    toc.append(
                        TocEntry(
                            level=depth,
                            title=parts[depth - 1],
                            annotation=(
                                f"{ds['files']} files | {format_size(ds['size'])}"
                            ),
                        )
                    )

        return toc[:30]  # Limit to 30 entries
