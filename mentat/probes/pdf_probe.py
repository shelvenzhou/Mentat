import fitz  # PyMuPDF
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import BaseProbe, ProbeResult


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
        for i in range(doc.page_count):  # Sample first pages
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
        return font_counts.most_common(1)[0][
            0
        ]  # Return the most common font size (body text)

    def extract_visual_structure(self, doc):
        """Extract structure based on visual information."""
        body_font_size = self.get_font_histogram(doc)
        candidates_toc = []
        captions = []
        inferred_title = None
        max_weighted_size = 0

        # Regex for Captions (e.g., Figure 1:, Table 1-1)
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
                    continue  # Ignore image blocks

                # Analyze inner spans for text and styling
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

                        # Check for bold flag (bit 4, value 16)
                        # flags & 16 != 0 means bold
                        if span["flags"] & 16:
                            bold_text_len += span_len

                        if span["size"] > max_span_size:
                            max_span_size = span["size"]

                block_text = "".join(block_text_parts).strip()
                if not block_text:
                    continue

                # Heuristic: If > 80% of text is bold, treat as a bold block
                is_bold_block = (
                    (bold_text_len / total_text_len) > BOLD_RATIO_THRESHOLD
                    if total_text_len > 0
                    else False
                )

                # Calculate weighted size: Bold blocks get a 20% boost
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
                # Simple logic: Starts with Fig/Table, and font size usually close to or smaller than body text
                # We use raw size for captions, not weighted
                if caption_pattern.match(block_text):
                    captions.append(
                        {
                            "page": i + 1,
                            "text": block_text,
                        }
                    )

                # 3. Find Potential Headers (build pseudo ToC)
                # Logic: Weighted size significantly larger than body text + moderate length
                if (
                    weighted_size > body_font_size * HEADER_SIZE_MULTIPLIER
                    and len(block_text) < MAX_HEADER_LENGTH
                    and len(block_text) > MIN_HEADER_LENGTH
                ):
                    candidates_toc.append(
                        {"level": "header", "title": block_text, "page": i + 1}
                    )

        return inferred_title, candidates_toc, captions

    def run(self, file_path: str) -> ProbeResult:
        doc = fitz.open(file_path)

        # Basic metadata
        metadata = doc.metadata
        toc = doc.get_toc()

        # --- Enhanced Part ---
        vis_title, vis_toc, captions = self.extract_visual_structure(doc)

        # Merge strategy: Prioritize metadata, fallback to visual inference
        final_title = metadata.get("title")
        if not final_title or final_title.strip() == "":
            final_title = vis_title

        final_toc = toc if toc else vis_toc  # Use visual inference if no native ToC

        # Enhance structure with captions
        structure = {
            "page_count": doc.page_count,
            "title": final_title,
            "authors": metadata.get("author"),
            "creation_date": metadata.get("creationDate"),
            "toc_source": "metadata" if toc else "visual_inference",
            "toc": [
                {
                    "title": item[1] if isinstance(item, list) else item["title"],
                    "page": item[2] if isinstance(item, list) else item["page"],
                }
                for item in final_toc
            ],
            "captions": captions,
            "is_encrypted": doc.is_encrypted,
            "metadata": metadata,
        }

        summary_hint = f"PDF: {final_title or 'Untitled'}. "
        summary_hint += (
            f"Found {len(structure['toc'])} sections and {len(captions)} captions."
        )

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="pdf",
            structure=structure,
            stats={},
            summary_hint=summary_hint,
        )
