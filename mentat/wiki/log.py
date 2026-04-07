"""Append-only wiki event log with simple file locking."""

from __future__ import annotations

import fcntl
import time
from pathlib import Path
from typing import Any


class WikiLog:
    """Append-only log used as the wiki work queue."""

    def __init__(self, wiki_dir: str | Path):
        self._wiki_dir = Path(wiki_dir)
        self._path = self._wiki_dir / "log.md"
        self._wiki_dir.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text(
                "# Mentat Wiki Log\n\n"
                "_Append-only timeline of ingest, sync, verify, and lint events._\n",
                "utf-8",
            )

    @property
    def path(self) -> Path:
        return self._path

    def append_event(self, event_type: str, **kv: Any) -> str:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        parts = [f"## [{timestamp}] {event_type}"]
        filename = kv.pop("filename", None)
        details = kv.pop("details", None)

        if filename:
            parts.append(str(filename))

        for key, value in kv.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple, set)):
                value = ",".join(str(item) for item in value)
            parts.append(f"{key}={value}")

        line = " | ".join(parts)

        with self._path.open("a", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.write(f"{line}\n")
            if details:
                if isinstance(details, str):
                    detail_lines = [details]
                else:
                    detail_lines = [str(item) for item in details]
                for detail_line in detail_lines:
                    handle.write(f"  {detail_line}\n")
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        return line
