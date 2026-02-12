import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes._utils import estimate_tokens, should_bypass, safe_read_text

# Common timestamp patterns
_TIMESTAMP_PATTERNS = [
    # ISO 8601: 2024-01-15T10:30:45 or 2024-01-15 10:30:45
    re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"),
    # Syslog: Jan 15 10:30:45
    re.compile(r"[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"),
    # Apache CLF: [15/Jan/2024:10:30:45 +0000]
    re.compile(r"\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}"),
    # Epoch-prefixed: 1705312245
    re.compile(r"^\d{10}"),
]

_LEVEL_RE = re.compile(
    r"\b(FATAL|CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG|TRACE)\b", re.IGNORECASE
)

# Simple stop words for keyword extraction
_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would shall should may might can could of in to for on with at by "
    "from as into through during before after above below between out off "
    "over under again further then once and but or nor not so yet both each "
    "few more most other some such no nor only own same than too very it its "
    "this that these those he she they them their we you your".split()
)


class LogProbe(BaseProbe):
    """Probe for log files."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".log")

    def run(self, file_path: str) -> ProbeResult:
        content = safe_read_text(file_path)
        lines = content.split("\n")
        non_empty = [l for l in lines if l.strip()]
        approx_tokens = estimate_tokens(content)

        # --- Log format detection ---
        log_format = self._detect_format(non_empty[:10])

        # --- Time range ---
        first_ts, last_ts = self._extract_time_range(non_empty)

        # --- Level statistics ---
        level_counts = self._count_levels(content)

        # --- Top keywords ---
        top_keywords = self._extract_keywords(non_empty)

        # --- Stats ---
        stats: Dict[str, Any] = {
            "line_count": len(non_empty),
            "approx_tokens": approx_tokens,
            "log_format": log_format,
            "level_counts": level_counts,
            "top_keywords": top_keywords,
        }
        if first_ts or last_ts:
            stats["time_range"] = {"first": first_ts, "last": last_ts}

        # --- Topic ---
        total_errors = level_counts.get("ERROR", 0) + level_counts.get("FATAL", 0)
        time_desc = ""
        if first_ts and last_ts:
            time_desc = f" spanning {first_ts} to {last_ts}"
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=(
                f"{log_format}-format log file with {len(non_empty)} lines{time_desc}"
                + (f", {total_errors} errors" if total_errors else "")
            ),
        )

        # --- ToC: level breakdown ---
        toc_entries: List[TocEntry] = []
        for level_name in ("FATAL", "CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "TRACE"):
            count = level_counts.get(level_name, 0)
            if count > 0:
                toc_entries.append(
                    TocEntry(
                        level=1,
                        title=level_name,
                        annotation=f"{count} occurrences",
                    )
                )

        structure = StructureInfo(toc=toc_entries)

        # --- Small-file bypass ---
        if should_bypass(content):
            stats["is_full_content"] = True
            return ProbeResult(
                filename=Path(file_path).name,
                file_type="log",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[Chunk(content=content, index=0)],
                raw_snippet=content,
            )

        # --- Sample chunks ---
        stats["is_full_content"] = False
        chunks = self._build_sample_chunks(non_empty, level_counts)

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="log",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet="\n".join(non_empty[:5]),
        )

    def _detect_format(self, sample_lines: List[str]) -> str:
        if not sample_lines:
            return "unknown"

        # Check if JSON-lines
        json_count = 0
        for line in sample_lines:
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                json_count += 1
        if json_count > len(sample_lines) * 0.7:
            return "jsonl"

        # Check for syslog pattern
        syslog_re = re.compile(r"^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+")
        if sum(1 for l in sample_lines if syslog_re.match(l)) > len(sample_lines) * 0.5:
            return "syslog"

        # Check for Apache CLF
        apache_re = re.compile(r'^\S+\s+\S+\s+\S+\s+\[.+\]\s+"[A-Z]+\s+')
        if sum(1 for l in sample_lines if apache_re.match(l)) > len(sample_lines) * 0.5:
            return "apache_clf"

        # Check for ISO timestamp prefix
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}")
        if sum(1 for l in sample_lines if iso_re.match(l)) > len(sample_lines) * 0.5:
            return "iso_timestamp"

        return "custom"

    def _extract_time_range(
        self, lines: List[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        first_ts = None
        last_ts = None
        for pattern in _TIMESTAMP_PATTERNS:
            # Check first few lines for first timestamp
            for line in lines[:20]:
                m = pattern.search(line)
                if m:
                    first_ts = m.group()
                    break
            # Check last few lines for last timestamp
            for line in reversed(lines[-20:]):
                m = pattern.search(line)
                if m:
                    last_ts = m.group()
                    break
            if first_ts:
                break
        return first_ts, last_ts

    def _count_levels(self, content: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for match in _LEVEL_RE.finditer(content):
            level = match.group(1).upper()
            # Normalize WARNING -> WARN
            if level == "WARNING":
                level = "WARN"
            counts[level] = counts.get(level, 0) + 1
        return counts

    def _extract_keywords(self, lines: List[str], top_n: int = 10) -> List[str]:
        # Strip timestamps and level markers
        words: List[str] = []
        for line in lines:
            # Remove timestamps
            cleaned = re.sub(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*", "", line)
            cleaned = _LEVEL_RE.sub("", cleaned)
            # Tokenize: only alpha words > 2 chars
            for word in re.findall(r"[a-zA-Z_]\w{2,}", cleaned):
                lower = word.lower()
                if lower not in _STOP_WORDS:
                    words.append(lower)

        counter = Counter(words)
        return [w for w, _ in counter.most_common(top_n)]

    def _build_sample_chunks(
        self, lines: List[str], level_counts: Dict[str, int]
    ) -> List[Chunk]:
        chunks: List[Chunk] = []

        # Head: first 5 lines
        head_lines = lines[:5]
        if head_lines:
            chunks.append(
                Chunk(content="\n".join(head_lines), index=0, section="head")
            )

        # Tail: last 5 lines
        tail_lines = lines[-5:]
        if tail_lines and len(lines) > 5:
            chunks.append(
                Chunk(content="\n".join(tail_lines), index=1, section="tail")
            )

        # Error sample: up to 5 error lines
        if level_counts.get("ERROR", 0) > 0 or level_counts.get("FATAL", 0) > 0:
            error_re = re.compile(r"\b(ERROR|FATAL|CRITICAL)\b", re.IGNORECASE)
            error_lines = [l for l in lines if error_re.search(l)][:5]
            if error_lines:
                chunks.append(
                    Chunk(
                        content="\n".join(error_lines),
                        index=len(chunks),
                        section="error_sample",
                    )
                )

        return chunks or [Chunk(content="\n".join(lines[:10]), index=0)]
