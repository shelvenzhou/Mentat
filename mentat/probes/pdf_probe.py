import fitz  # PyMuPDF
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Caption,
    Chunk,
)
from mentat.probes.instruction_templates import (
    PDF_BRIEF_INTRO,
    PDF_INSTRUCTIONS,
    PDF_TOC_METHOD_METADATA,
    PDF_TOC_METHOD_VISUAL,
    PDF_TOC_SOURCE_METADATA,
    PDF_TOC_SOURCE_VISUAL,
)


BOLD_RATIO_THRESHOLD = 0.8
BOLD_WEIGHT_MULTIPLIER = 2

MAX_TITLE_LENGTH = 100
HEADER_SIZE_MULTIPLIER = 1.1
MAX_HEADER_LENGTH = 100
MIN_HEADER_LENGTH = 2


class PDFProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".pdf") or content_type == "application/pdf"

    def get_font_histogram(self, doc):
        """Calculate font size distribution across the document to find the body text font size."""
        font_counts = Counter()
        for i in range(doc.page_count):
            page = doc.load_page(i)
            try:
                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if b["type"] == 0:  # text
                        for line in b["lines"]:
                            for span in line["spans"]:
                                font_counts[round(span["size"], 1)] += len(span["text"])
            except Exception:
                continue

        if not font_counts:
            return 11.0  # default
        return font_counts.most_common(1)[0][0]

    def extract_visual_structure(self, doc):
        """Extract structure based on visual information (font size, bold, captions)."""
        body_font_size = self.get_font_histogram(doc)
        candidates_toc = []
        captions = []
        inferred_title = None
        max_weighted_size = 0

        caption_pattern = re.compile(
            r"^(Figure|Table|Fig\.|Tab\.)\s*\d+", re.IGNORECASE
        )

        for i, page in enumerate(doc):
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue

            for block in blocks:
                if block["type"] != 0:
                    continue

                block_text_parts = []
                max_span_size = 0
                total_text_len = 0
                bold_text_len = 0

                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"]
                        if not text.strip():
                            continue

                        block_text_parts.append(text)
                        span_len = len(text)
                        total_text_len += span_len

                        if span["flags"] & 16:
                            bold_text_len += span_len

                        if span["size"] > max_span_size:
                            max_span_size = span["size"]

                block_text = "".join(block_text_parts).strip()
                if not block_text:
                    continue

                is_bold_block = (
                    (bold_text_len / total_text_len) > BOLD_RATIO_THRESHOLD
                    if total_text_len > 0
                    else False
                )

                weighted_size = (
                    max_span_size * BOLD_WEIGHT_MULTIPLIER
                    if is_bold_block
                    else max_span_size
                )

                # 1. Find Title (first page only)
                if i == 0:
                    if (
                        weighted_size > max_weighted_size
                        and len(block_text) < MAX_TITLE_LENGTH
                    ):
                        max_weighted_size = weighted_size
                        inferred_title = block_text

                # 2. Find Captions
                if caption_pattern.match(block_text):
                    kind = (
                        "table"
                        if block_text.lower().startswith(("table", "tab."))
                        else "figure"
                    )
                    captions.append(
                        Caption(
                            text=block_text,
                            page=i + 1,
                            kind=kind,
                        )
                    )

                # 3. Find Potential Headers (build pseudo ToC)
                if (
                    weighted_size > body_font_size * HEADER_SIZE_MULTIPLIER
                    and len(block_text) < MAX_HEADER_LENGTH
                    and len(block_text) > MIN_HEADER_LENGTH
                ):
                    candidates_toc.append(
                        TocEntry(level=1, title=block_text, page=i + 1)
                    )

        return inferred_title, candidates_toc, captions

    def _extract_first_paragraph(self, doc) -> str:
        """Extract the first meaningful paragraph of body text."""
        body_font_size = self.get_font_histogram(doc)
        for page in doc:
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception:
                continue
            for block in blocks:
                if block["type"] != 0:
                    continue
                text_parts = []
                for line in block["lines"]:
                    for span in line["spans"]:
                        if abs(span["size"] - body_font_size) < 1.0:
                            text_parts.append(span["text"])
                text = "".join(text_parts).strip()
                if len(text) > 80:  # Skip short fragments
                    return text[:500]
        return ""

    def _add_previews(self, doc, toc_entries: List[TocEntry]) -> None:
        """Populate preview (first sentence after heading) for each ToC entry."""
        for entry in toc_entries:
            if not entry.page or entry.preview:
                continue
            page_idx = entry.page - 1
            if page_idx < 0 or page_idx >= doc.page_count:
                continue
            page_text = doc.load_page(page_idx).get_text("text")
            # Find the title text on its page
            title_pos = page_text.find(entry.title)
            if title_pos < 0:
                continue
            after = page_text[title_pos + len(entry.title) :].strip()
            if not after:
                continue
            # First sentence (up to '. ' or first 200 chars)
            m = re.search(r"[.!?]\s", after[:300])
            if m:
                entry.preview = after[: m.end()].strip()
            elif len(after) > 20:
                entry.preview = after[:200].strip()

    def _build_chunks(self, doc, toc_entries: List[TocEntry]) -> List[Chunk]:
        """Build format-aware chunks: one chunk per page, tagged with the current section."""
        chunks = []
        # Build a page -> section mapping from ToC
        section_map = {}
        current_section = None
        for entry in toc_entries:
            if entry.page:
                current_section = entry.title
                section_map[entry.page] = current_section

        current_section = None
        for i, page in enumerate(doc):
            page_num = i + 1
            if page_num in section_map:
                current_section = section_map[page_num]
            text = page.get_text("text").strip()
            if text:
                chunks.append(
                    Chunk(
                        content=text,
                        index=i,
                        section=current_section,
                        page=page_num,
                    )
                )
        return chunks

    def run(self, file_path: str) -> ProbeResult:
        doc = fitz.open(file_path)

        # Basic metadata
        metadata = doc.metadata
        toc = doc.get_toc()

        # Visual structure extraction
        vis_title, vis_toc, captions = self.extract_visual_structure(doc)

        # Merge: Prioritize metadata, fallback to visual inference
        final_title = metadata.get("title")
        if not final_title or final_title.strip() == "":
            final_title = vis_title

        final_toc_entries = []
        if toc:
            for item in toc:
                final_toc_entries.append(
                    TocEntry(level=item[0], title=item[1], page=item[2])
                )
        else:
            final_toc_entries = vis_toc

        # Add previews to ToC entries
        self._add_previews(doc, final_toc_entries)

        # Extract first paragraph for topic
        first_para = self._extract_first_paragraph(doc)

        # Build chunks
        chunks = self._build_chunks(doc, final_toc_entries)

        topic = TopicInfo(
            title=final_title,
            first_paragraph=first_para,
        )

        structure = StructureInfo(
            toc=final_toc_entries,
            captions=captions,
        )

        stats = {
            "page_count": doc.page_count,
            "toc_source": "metadata" if toc else "visual_inference",
            "is_encrypted": doc.is_encrypted,
            "authors": metadata.get("author"),
            "creation_date": metadata.get("creationDate"),
        }

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="pdf",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate PDF-specific instructions."""
        stats = probe_result.stats
        toc_source = stats.get('toc_source', 'visual_inference')

        # Brief intro
        toc_method = PDF_TOC_METHOD_METADATA if toc_source == 'metadata' else PDF_TOC_METHOD_VISUAL
        brief_intro = PDF_BRIEF_INTRO.format(toc_method=toc_method)

        # Full instructions
        toc_source_desc = PDF_TOC_SOURCE_METADATA if toc_source == 'metadata' else PDF_TOC_SOURCE_VISUAL
        instructions = PDF_INSTRUCTIONS.format(
            toc_source=toc_source_desc,
            filename=probe_result.filename,
        )

        return brief_intro, instructions
