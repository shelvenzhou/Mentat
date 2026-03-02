import re
from pathlib import Path
from typing import List, Optional, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes._utils import estimate_tokens, extract_preview, SMALL_FILE_TOKENS
from mentat.probes.instruction_templates import (
    MARKDOWN_BRIEF_INTRO,
    MARKDOWN_INSTRUCTIONS,
)

# If heading density exceeds this AND tokens < _DENSITY_TOKEN_CAP, return full content.
_HEADING_DENSITY_THRESHOLD = 0.25
_DENSITY_TOKEN_CAP = 3000

_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_LINK_RE = re.compile(r"\[.*?\]\(.*?\)")
_HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _code_fence_ranges(content: str) -> List[Tuple[int, int]]:
    """Return (start, end) character ranges for fenced code blocks."""
    ranges = []
    it = _CODE_FENCE_RE.finditer(content)
    for m in it:
        open_pos = m.start()
        close = next(it, None)
        if close:
            ranges.append((open_pos, close.end()))
        else:
            # Unclosed fence: treat rest of file as code
            ranges.append((open_pos, len(content)))
    return ranges


def _filter_headers(content: str, header_matches: list) -> list:
    """Remove header matches that fall inside fenced code blocks."""
    ranges = _code_fence_ranges(content)
    if not ranges:
        return header_matches
    return [
        m for m in header_matches
        if not any(start <= m.start() < end for start, end in ranges)
    ]


def _annotate_section(section_body: str) -> Optional[str]:
    """Detect structural features within a section body and return a compact annotation."""
    parts = []

    list_items = len(_LIST_ITEM_RE.findall(section_body))
    if list_items:
        parts.append(f"List, {list_items} items")

    code_fences = len(_CODE_FENCE_RE.findall(section_body))
    code_blocks = code_fences // 2
    if code_blocks:
        parts.append(f"Code blocks, {code_blocks}")

    links = len(_LINK_RE.findall(section_body))
    if links:
        parts.append(f"{links} links")

    line_count = len([l for l in section_body.split("\n") if l.strip()])
    parts.append(f"{line_count} lines")

    return " | ".join(parts) if parts else None


class MarkdownProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".md", ".markdown"))
            or content_type == "text/markdown"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")
        words = content.split()
        approx_tokens = estimate_tokens(content)

        # --- Collect raw header matches, excluding code blocks ---
        header_matches = _filter_headers(content, list(_HEADER_RE.finditer(content)))

        # --- Basic stats (computed regardless of path) ---
        links = _LINK_RE.findall(content)
        code_fences = _CODE_FENCE_RE.findall(content)
        total_lines = len(lines)
        heading_density = len(header_matches) / max(total_lines, 1)

        stats = {
            "link_count": len(links),
            "code_block_count": len(code_fences) // 2,
            "word_count": len(words),
            "approx_tokens": approx_tokens,
            "heading_density": round(heading_density, 3),
        }

        # --- Topic: title from first H1, first paragraph ---
        topic = self._extract_topic(lines, header_matches, file_path)

        # --- Decision: should we return full content? ---
        is_small = approx_tokens < SMALL_FILE_TOKENS
        is_fragmented = (
            heading_density > _HEADING_DENSITY_THRESHOLD
            and approx_tokens < _DENSITY_TOKEN_CAP
        )

        if is_small or is_fragmented:
            stats["is_full_content"] = True
            # Build enhanced ToC even for small files — preview/annotation
            # are valuable for toc_only search and get_summary, where the
            # full content is NOT returned.
            if header_matches:
                toc_entries, _ = self._build_enhanced_toc(content, header_matches)
            else:
                toc_entries = []
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type="markdown",
                topic=topic,
                structure=StructureInfo(toc=toc_entries),
                stats=stats,
                chunks=[Chunk(content=content, index=0)],
                raw_snippet=content,
            )
            brief_intro, instructions = self.generate_instructions(result)
            result.brief_intro = brief_intro
            result.instructions = instructions
            return result

        # --- Enhanced skeleton path ---
        stats["is_full_content"] = False
        toc_entries, _ = self._build_enhanced_toc(
            content, header_matches
        )

        # --- Chunks: split by headers with enriched section names ---
        chunks = self._split_by_headers(content, header_matches)

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="markdown",
            topic=topic,
            structure=StructureInfo(toc=toc_entries),
            stats=stats,
            chunks=chunks,
            raw_snippet=content[:500],
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate Markdown-specific instructions."""
        # Brief intro
        brief_intro = MARKDOWN_BRIEF_INTRO

        # Full instructions
        instructions = MARKDOWN_INSTRUCTIONS.format(filename=probe_result.filename)

        return brief_intro, instructions

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _extract_topic(
        self,
        lines: List[str],
        header_matches: list,
        file_path: str,
    ) -> TopicInfo:
        title = None
        for m in header_matches:
            if len(m.group(1)) == 1:
                title = m.group(2).strip()
                break

        # First paragraph: first non-empty, non-header block
        para_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                para_lines.append(stripped)
                if len(" ".join(para_lines)) > 100:
                    break
            elif para_lines:
                break

        first_paragraph = " ".join(para_lines) if para_lines else None

        return TopicInfo(
            title=title or Path(file_path).stem,
            first_paragraph=first_paragraph,
        )

    def _build_enhanced_toc(
        self,
        content: str,
        header_matches: list,
    ) -> Tuple[List[TocEntry], List[str]]:
        """Build ToC with preview and annotation for each section."""
        entries: List[TocEntry] = []
        section_bodies: List[str] = []

        for i, match in enumerate(header_matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Section body = text between this header and the next (or EOF)
            body_start = match.end()
            body_end = (
                header_matches[i + 1].start()
                if i + 1 < len(header_matches)
                else len(content)
            )
            body = content[body_start:body_end]
            section_bodies.append(body)

            preview = extract_preview(body)
            annotation = _annotate_section(body)

            entries.append(
                TocEntry(
                    level=level,
                    title=title,
                    preview=preview,
                    annotation=annotation,
                )
            )

        return entries, section_bodies

    def _split_by_headers(
        self, content: str, header_matches: list
    ) -> List[Chunk]:
        """Split content by headers, then merge small adjacent chunks."""
        if not header_matches:
            return [Chunk(content=content, index=0)]

        chunks: List[Chunk] = []

        # Content before first header
        pre = content[: header_matches[0].start()].strip()
        if pre:
            chunks.append(Chunk(content=pre, index=0, section="preamble", metadata={"level": 0}))

        for i, match in enumerate(header_matches):
            start = match.start()
            end = (
                header_matches[i + 1].start()
                if i + 1 < len(header_matches)
                else len(content)
            )
            section_text = content[start:end].strip()
            section_title = match.group(2).strip()
            level = len(match.group(1))  # number of # chars
            if section_text:
                chunks.append(
                    Chunk(
                        content=section_text,
                        index=len(chunks),
                        section=section_title,
                        metadata={"level": level},
                    )
                )

        return chunks
