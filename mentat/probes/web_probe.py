import re
import json
import trafilatura
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
from mentat.probes._utils import estimate_tokens, should_bypass, extract_preview
from mentat.librarian.instruction_templates import (
    WEB_BRIEF_INTRO,
    WEB_INSTRUCTIONS,
)

# Regex patterns for HTML structure extraction
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
_META_DESC_RE = re.compile(
    r'<meta\s+[^>]*name=["\']description["\'][^>]*content=["\'](.*?)["\']',
    re.IGNORECASE,
)
_META_DESC_RE2 = re.compile(
    r'<meta\s+[^>]*content=["\'](.*?)["\'][^>]*name=["\']description["\']',
    re.IGNORECASE,
)
_META_KEYWORDS_RE = re.compile(
    r'<meta\s+[^>]*name=["\']keywords["\'][^>]*content=["\'](.*?)["\']',
    re.IGNORECASE,
)
_HEADING_RE = re.compile(
    r"<(h[1-6])[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL
)
_SEMANTIC_ELEMENTS = ("nav", "header", "main", "article", "section", "footer", "aside")
_SEMANTIC_RE = re.compile(
    r"<(" + "|".join(_SEMANTIC_ELEMENTS) + r")[\s>]", re.IGNORECASE
)
_NAV_LINK_RE = re.compile(
    r"<nav[^>]*>(.*?)</nav>", re.IGNORECASE | re.DOTALL
)
_LINK_RE = re.compile(r"<a\s+[^>]*href=[\"'](.*?)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.MULTILINE)


def _strip_tags(html: str) -> str:
    """Remove HTML tags, returning plain text."""
    return _TAG_RE.sub("", html).strip()


class WebProbe(BaseProbe):
    """Probe for HTML/Web pages."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".html", ".htm")) or content_type == "text/html"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # ========================================
        # Pass 1: Raw HTML structure extraction
        # ========================================

        # Title
        html_title = None
        m = _TITLE_RE.search(html_content)
        if m:
            html_title = _strip_tags(m.group(1)).strip()

        # Meta description
        meta_desc = None
        m = _META_DESC_RE.search(html_content) or _META_DESC_RE2.search(html_content)
        if m:
            meta_desc = m.group(1).strip()

        # Meta keywords
        meta_keywords = None
        m = _META_KEYWORDS_RE.search(html_content)
        if m:
            meta_keywords = m.group(1).strip()

        # Heading hierarchy
        heading_matches = list(_HEADING_RE.finditer(html_content))

        # Semantic elements present
        semantic_found = list(set(m.group(1).lower() for m in _SEMANTIC_RE.finditer(html_content)))

        # Nav links
        nav_links: List[str] = []
        nav_match = _NAV_LINK_RE.search(html_content)
        if nav_match:
            for link_match in _LINK_RE.finditer(nav_match.group(1)):
                link_text = _strip_tags(link_match.group(2)).strip()
                if link_text:
                    nav_links.append(link_text)

        # ========================================
        # Pass 2: Trafilatura for clean text
        # ========================================
        extracted_data = trafilatura.extract(
            html_content,
            output_format="json",
            include_comments=False,
            include_tables=True,
        )
        if extracted_data:
            data = json.loads(extracted_data)
        else:
            data = {"title": None, "author": None, "date": None, "text": ""}

        content_text = data.get("text", "")
        approx_tokens = estimate_tokens(content_text) if content_text else 0

        # ========================================
        # Build ToC from heading hierarchy
        # ========================================
        toc_entries: List[TocEntry] = []

        # Build section bodies for preview/annotation
        for i, hm in enumerate(heading_matches):
            level = int(hm.group(1)[1])  # h1 -> 1, h2 -> 2, etc.
            title = _strip_tags(hm.group(2)).strip()
            if not title:
                continue

            # Section body: HTML between this heading and the next
            body_start = hm.end()
            body_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(html_content)
            section_html = html_content[body_start:body_end]
            section_text = _strip_tags(section_html)

            preview = extract_preview(section_text) if section_text.strip() else None
            annotation = self._annotate_html_section(section_html)

            toc_entries.append(
                TocEntry(
                    level=level,
                    title=title,
                    preview=preview,
                    annotation=annotation,
                )
            )

        # ========================================
        # Build result
        # ========================================

        # Topic
        topic = TopicInfo(
            title=html_title or data.get("title") or Path(file_path).stem,
            abstract=meta_desc,
            first_paragraph=content_text[:500] if content_text else None,
        )

        structure = StructureInfo(toc=toc_entries)

        # Stats
        stats: Dict[str, Any] = {
            "hostname": data.get("hostname"),
            "author": data.get("author"),
            "date": data.get("date"),
            "categories": data.get("categories"),
            "tags": data.get("tags"),
            "content_length": len(content_text),
            "word_count": len(content_text.split()),
            "approx_tokens": approx_tokens,
            "heading_count": len(toc_entries),
            "semantic_elements": semantic_found,
            "nav_link_count": len(nav_links),
        }
        if meta_keywords:
            stats["meta_keywords"] = meta_keywords

        # Small-file bypass
        if content_text and should_bypass(content_text):
            stats["is_full_content"] = True
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type="web",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[Chunk(content=content_text, index=0)],
                raw_snippet=content_text,
            )
            brief_intro, instructions = self.generate_instructions(result)
            result.brief_intro = brief_intro
            result.instructions = instructions
            return result

        # Chunks: split by heading sections
        stats["is_full_content"] = False
        chunks = self._build_section_chunks(content_text, toc_entries, heading_matches, html_content)

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="web",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=content_text[:500] if content_text else None,
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate Web/HTML-specific instructions."""
        # Brief intro
        brief_intro = WEB_BRIEF_INTRO

        # Full instructions
        instructions = WEB_INSTRUCTIONS.format(filename=probe_result.filename)

        return brief_intro, instructions

    def _annotate_html_section(self, section_html: str) -> Optional[str]:
        """Annotate an HTML section with structural features."""
        parts = []

        # Count paragraphs
        p_count = section_html.lower().count("<p")
        if p_count:
            parts.append(f"{p_count} paragraphs")

        # Count links
        link_count = len(_LINK_RE.findall(section_html))
        if link_count:
            parts.append(f"{link_count} links")

        # Count list items
        li_count = section_html.lower().count("<li")
        if li_count:
            parts.append(f"List, {li_count} items")

        # Count images
        img_count = section_html.lower().count("<img")
        if img_count:
            parts.append(f"{img_count} images")

        # Count tables
        table_count = section_html.lower().count("<table")
        if table_count:
            parts.append(f"{table_count} tables")

        # Line count of plain text
        text = _strip_tags(section_html)
        non_empty = len([l for l in text.split("\n") if l.strip()])
        if non_empty:
            parts.append(f"{non_empty} lines")

        return " | ".join(parts) if parts else None

    def _build_section_chunks(
        self,
        content_text: str,
        toc_entries: List[TocEntry],
        heading_matches: list,
        html_content: str,
    ) -> List[Chunk]:
        """Build chunks by splitting extracted text at heading boundaries."""
        if not content_text:
            return []

        if not toc_entries:
            # No headings found: fall back to paragraph splitting
            paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
            return [
                Chunk(content=para, index=i)
                for i, para in enumerate(paragraphs)
            ]

        # Map heading titles to split positions in the extracted text
        chunks: List[Chunk] = []

        # Build title → level map from toc_entries
        title_level = {e.title: e.level for e in toc_entries}

        # Try to split the extracted text by finding heading titles within it
        positions = []
        for entry in toc_entries:
            idx = content_text.find(entry.title)
            if idx >= 0:
                positions.append((idx, entry.title, entry.level))

        positions.sort()

        if not positions:
            # Titles not found in extracted text: fall back to paragraph splitting
            paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
            return [
                Chunk(content=para, index=i)
                for i, para in enumerate(paragraphs)
            ]

        # Pre-heading content
        if positions[0][0] > 0:
            pre = content_text[: positions[0][0]].strip()
            if pre:
                chunks.append(Chunk(content=pre, index=0, section="preamble", metadata={"level": 0}))

        for i, (pos, title, level) in enumerate(positions):
            end = positions[i + 1][0] if i + 1 < len(positions) else len(content_text)
            section_text = content_text[pos:end].strip()
            if section_text:
                chunks.append(
                    Chunk(
                        content=section_text,
                        index=len(chunks),
                        section=title,
                        metadata={"level": level},
                    )
                )

        chunks = chunks or [Chunk(content=content_text, index=0)]
        return chunks
