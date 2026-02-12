"""Shared utilities for all probes."""

from typing import Optional

# Threshold below which probes should return full content instead of a skeleton.
SMALL_FILE_TOKENS = 1000


def estimate_tokens(text: str) -> int:
    """Estimate token count from text (word-count * 1.3)."""
    return int(len(text.split()) * 1.3)


def should_bypass(text: str, threshold: int = SMALL_FILE_TOKENS) -> bool:
    """Return True if the file is small enough to return full content."""
    return estimate_tokens(text) < threshold


def extract_preview(text: str, max_len: int = 120) -> Optional[str]:
    """Extract the first meaningful non-empty line from a text block.

    Skips blank lines and markdown headers. Truncates at word boundary
    if the line exceeds max_len.
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            if len(stripped) > max_len:
                cut = stripped[:max_len].rsplit(" ", 1)[0]
                return cut + "..."
            return stripped
    return None


def safe_read_text(file_path: str, encoding: str = "utf-8") -> str:
    """Read a text file with fallback to latin-1 if utf-8 fails."""
    try:
        with open(file_path, "r", encoding=encoding) as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def truncate_string(s: str, head: int = 10, tail: int = 10) -> str:
    """Truncate a long string to head...tail format with length annotation."""
    if len(s) <= head + tail + 5:
        return s
    return f"{s[:head]}...{s[-tail:]} (Len: {len(s)})"


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
