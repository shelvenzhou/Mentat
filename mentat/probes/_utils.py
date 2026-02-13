"""Shared utilities for all probes."""

from typing import Any, Dict, FrozenSet, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mentat.probes.base import Chunk

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


# ---------------------------------------------------------------------------
# Chunk merging
# ---------------------------------------------------------------------------

MIN_CHUNK_TOKENS = 300  # Chunks below this are candidates for merging
MAX_CHUNK_TOKENS = 1200 # Merged chunk should not exceed this


def merge_small_chunks(
    chunks: List["Chunk"],
    min_tokens: int = MIN_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    hard_boundary_levels: FrozenSet[int] = frozenset({1, 2}),
) -> List["Chunk"]:
    """Merge adjacent small chunks, respecting hierarchical boundaries.

    After structural splitting, probes often produce many tiny chunks
    (one per heading section).  This merges adjacent chunks whose token
    count is below *min_tokens*, provided:

    1. The merged result would not exceed *max_tokens*.
    2. Neither chunk sits at a "hard boundary" level (default: H1/H2).

    Hierarchy is read from ``chunk.metadata["level"]``.  Chunks without
    a level (e.g. JSON) use purely size-based merging.

    Returns a new list with fresh sequential ``index`` values.
    """
    from mentat.probes.base import Chunk

    if len(chunks) <= 1:
        return list(chunks)

    merged: List[Chunk] = []

    # Accumulator state
    acc_parts: List[str] = []
    acc_tokens: int = 0
    acc_section: Optional[str] = None
    acc_page: Optional[int] = None
    acc_meta: Dict[str, Any] = {}

    def _flush() -> None:
        nonlocal acc_parts, acc_tokens, acc_section, acc_page, acc_meta
        if acc_parts:
            merged.append(
                Chunk(
                    content="\n\n".join(acc_parts),
                    index=len(merged),
                    section=acc_section,
                    page=acc_page,
                    metadata=acc_meta,
                )
            )
            acc_parts = []
            acc_tokens = 0
            acc_section = None
            acc_page = None
            acc_meta = {}

    def _is_hard(chunk: "Chunk") -> bool:
        level = chunk.metadata.get("level")
        return level is not None and level in hard_boundary_levels

    for chunk in chunks:
        tok = estimate_tokens(chunk.content)

        # Hard boundary: always flush previous, start fresh
        if _is_hard(chunk):
            _flush()
            acc_parts = [chunk.content]
            acc_tokens = tok
            acc_section = chunk.section
            acc_page = chunk.page
            acc_meta = dict(chunk.metadata)
            # If this chunk is already large enough, flush it immediately
            if tok >= min_tokens:
                _flush()
            continue

        # Empty accumulator: start a new group
        if not acc_parts:
            acc_parts = [chunk.content]
            acc_tokens = tok
            acc_section = chunk.section
            acc_page = chunk.page
            acc_meta = dict(chunk.metadata)
            continue

        combined = acc_tokens + tok

        # Accumulator is small — try to pull in more content
        if acc_tokens < min_tokens and combined <= max_tokens:
            acc_parts.append(chunk.content)
            acc_tokens = combined
            continue

        # Current chunk is small — merge it into the accumulator if it fits
        if tok < min_tokens and combined <= max_tokens:
            acc_parts.append(chunk.content)
            acc_tokens = combined
            continue

        # Can't merge: flush and start new group
        _flush()
        acc_parts = [chunk.content]
        acc_tokens = tok
        acc_section = chunk.section
        acc_page = chunk.page
        acc_meta = dict(chunk.metadata)

    _flush()
    return merged
