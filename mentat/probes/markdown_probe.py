import re
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)


class MarkdownProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".md", ".markdown"))
            or content_type == "text/markdown"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # --- Structure: Header hierarchy as ToC ---
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
        toc_entries = []
        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            toc_entries.append(TocEntry(level=level, title=title))

        structure = StructureInfo(toc=toc_entries)

        # --- Topic: title from first H1, first paragraph ---
        title = None
        first_paragraph = None
        for entry in toc_entries:
            if entry.level == 1:
                title = entry.title
                break

        # First paragraph: first non-empty, non-header block of text
        lines = content.split("\n")
        para_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                para_lines.append(stripped)
                if len(" ".join(para_lines)) > 100:
                    break
            elif para_lines:
                break
        if para_lines:
            first_paragraph = " ".join(para_lines)

        topic = TopicInfo(
            title=title or Path(file_path).stem, first_paragraph=first_paragraph
        )

        # --- Stats ---
        links = re.findall(r"\[.*?\]\(.*?\)", content)
        code_blocks = re.findall(r"```", content)
        words = content.split()

        stats = {
            "link_count": len(links),
            "code_block_count": len(code_blocks) // 2,
            "word_count": len(words),
            "approx_tokens": int(len(words) * 1.3),
        }

        # --- Chunks: split by top-level headers ---
        chunks = self._split_by_headers(content, toc_entries)

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="markdown",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=content[:500],
        )

    def _split_by_headers(
        self, content: str, toc_entries: List[TocEntry]
    ) -> List[Chunk]:
        """Split content by top-level headers so each chunk has section context."""
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
        matches = list(header_pattern.finditer(content))

        if not matches:
            return [Chunk(content=content, index=0)]

        chunks = []
        # Content before first header
        pre = content[: matches[0].start()].strip()
        if pre:
            chunks.append(Chunk(content=pre, index=0, section="preamble"))

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            section_text = content[start:end].strip()
            section_title = match.group(2).strip()
            if section_text:
                chunks.append(
                    Chunk(
                        content=section_text,
                        index=len(chunks),
                        section=section_title,
                    )
                )

        return chunks
