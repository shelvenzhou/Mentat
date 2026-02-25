"""Shared utilities for all probes."""

from typing import Any, Dict, FrozenSet, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mentat.probes.base import Chunk

# Threshold below which probes should return full content instead of a skeleton.
# Single files under this size skip skeleton extraction and return content directly.
SMALL_FILE_TOKENS = 2000


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


# ---------------------------------------------------------------------------
# Chunk normalization (merge small + split large)
# ---------------------------------------------------------------------------

CHUNK_TARGET_TOKENS = 1000  # Default target chunk size for retrieval
CHUNK_OVERLAP_TOKENS = 50   # Overlap between split pieces for context continuity
CHUNK_MIN_TOKENS = 100      # Chunks below this are candidates for merging


def _split_text(
    text: str,
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> List[str]:
    """Split text into fixed-size pieces at paragraph/sentence boundaries.

    Splitting priority:
      1. Paragraph boundaries (``\\n\\n``)
      2. Sentence boundaries (``. `` followed by uppercase or newline)
      3. Word boundaries (hard cut)

    Each piece (except the first) starts with *overlap_tokens* worth of
    trailing text from the previous piece to maintain context continuity.

    Returns a list of text pieces.
    """
    if estimate_tokens(text) <= target_tokens:
        return [text]

    # Split into paragraphs
    paragraphs = text.split("\n\n")

    pieces: List[str] = []
    current_parts: List[str] = []
    current_tokens: int = 0

    def _flush_current() -> None:
        nonlocal current_parts, current_tokens
        if current_parts:
            pieces.append("\n\n".join(current_parts))
            current_parts = []
            current_tokens = 0

    def _get_overlap_text(piece: str) -> str:
        """Extract last ~overlap_tokens worth of text from a piece."""
        if overlap_tokens <= 0:
            return ""
        words = piece.split()
        # Convert token target to approximate word count
        overlap_words = max(1, int(overlap_tokens / 1.3))
        if len(words) <= overlap_words:
            return piece
        return " ".join(words[-overlap_words:])

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = estimate_tokens(para)

        # Single paragraph exceeds target — split by sentences
        if para_tokens > target_tokens:
            _flush_current()
            sentence_pieces = _split_by_sentences(para, target_tokens)
            for sp in sentence_pieces:
                if pieces and overlap_tokens > 0:
                    overlap = _get_overlap_text(pieces[-1])
                    pieces.append(overlap + "\n" + sp)
                else:
                    pieces.append(sp)
            continue

        # Would adding this paragraph exceed target?
        if current_tokens + para_tokens > target_tokens and current_parts:
            _flush_current()
            # Start new piece with overlap from previous
            if pieces and overlap_tokens > 0:
                overlap = _get_overlap_text(pieces[-1])
                current_parts = [overlap]
                current_tokens = estimate_tokens(overlap)

        current_parts.append(para)
        current_tokens += para_tokens

    _flush_current()

    return pieces if pieces else [text]


def _split_by_sentences(
    text: str, target_tokens: int,
) -> List[str]:
    """Split a long paragraph into pieces at sentence boundaries.

    Falls back to word-level splitting for very long sentences.
    """
    import re
    # Split on sentence-ending punctuation followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text)

    pieces: List[str] = []
    current_parts: List[str] = []
    current_tokens: int = 0

    for sent in sentences:
        sent_tokens = estimate_tokens(sent)

        # Single sentence exceeds target — hard word split
        if sent_tokens > target_tokens:
            if current_parts:
                pieces.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            # Word-level splitting
            words = sent.split()
            word_target = max(1, int(target_tokens / 1.3))
            for i in range(0, len(words), word_target):
                pieces.append(" ".join(words[i : i + word_target]))
            continue

        if current_tokens + sent_tokens > target_tokens and current_parts:
            pieces.append(" ".join(current_parts))
            current_parts = []
            current_tokens = 0

        current_parts.append(sent)
        current_tokens += sent_tokens

    if current_parts:
        pieces.append(" ".join(current_parts))

    return pieces if pieces else [text]


def normalize_chunk_sizes(
    chunks: List["Chunk"],
    target_tokens: int = CHUNK_TARGET_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    min_chunk_tokens: int = CHUNK_MIN_TOKENS,
) -> List["Chunk"]:
    """Normalize chunk sizes: merge very small chunks, split oversized ones.

    Two-pass normalization that replaces ``merge_small_chunks`` as the
    standard post-processing step:

    **Pass 1 — Merge:** Adjacent chunks below *min_chunk_tokens* are merged
    as long as the result stays within *target_tokens*.

    **Pass 2 — Split:** Chunks exceeding *target_tokens* are split into
    fixed-size pieces at paragraph/sentence boundaries with *overlap_tokens*
    of overlap.  Each piece inherits the parent chunk's ``section``,
    ``page``, and ``metadata``.

    Returns a new list with fresh sequential ``index`` values.
    """
    from mentat.probes.base import Chunk

    if not chunks:
        return []

    # ── Pass 1: merge small adjacent chunks ──────────────────────────
    merged: List[Chunk] = []
    acc_parts: List[str] = []
    acc_tokens: int = 0
    acc_section: Optional[str] = None
    acc_page: Optional[int] = None
    acc_meta: Dict[str, Any] = {}

    def _flush_merge() -> None:
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

    for chunk in chunks:
        tok = estimate_tokens(chunk.content)

        if not acc_parts:
            # Start new group
            acc_parts = [chunk.content]
            acc_tokens = tok
            acc_section = chunk.section
            acc_page = chunk.page
            acc_meta = dict(chunk.metadata)
            continue

        combined = acc_tokens + tok

        # Merge if both current accumulator and chunk are small enough
        if acc_tokens < min_chunk_tokens and combined <= target_tokens:
            acc_parts.append(chunk.content)
            acc_tokens = combined
            continue

        if tok < min_chunk_tokens and combined <= target_tokens:
            acc_parts.append(chunk.content)
            acc_tokens = combined
            continue

        # Can't merge — flush and start new group
        _flush_merge()
        acc_parts = [chunk.content]
        acc_tokens = tok
        acc_section = chunk.section
        acc_page = chunk.page
        acc_meta = dict(chunk.metadata)

    _flush_merge()

    # ── Pass 2: split oversized chunks ───────────────────────────────
    result: List[Chunk] = []

    for chunk in merged:
        tok = estimate_tokens(chunk.content)
        if tok <= target_tokens:
            result.append(
                Chunk(
                    content=chunk.content,
                    index=len(result),
                    section=chunk.section,
                    page=chunk.page,
                    metadata=chunk.metadata,
                )
            )
        else:
            # Split into fixed-size pieces
            pieces = _split_text(chunk.content, target_tokens, overlap_tokens)
            for piece_text in pieces:
                result.append(
                    Chunk(
                        content=piece_text,
                        index=len(result),
                        section=chunk.section,
                        page=chunk.page,
                        metadata=dict(chunk.metadata),
                    )
                )

    return result
