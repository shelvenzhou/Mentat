import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import BaseProbe, ProbeResult


class PDFProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".pdf") or content_type == "application/pdf"

    def run(self, file_path: str) -> ProbeResult:
        doc = fitz.open(file_path)

        # 1. Structure: Table of Contents and Page Count
        toc = doc.get_toc()
        structure = {
            "page_count": doc.page_count,
            "toc": [
                {"level": item[0], "title": item[1], "page": item[2]} for item in toc
            ],
            "is_encrypted": doc.is_encrypted,
            "metadata": doc.metadata,
        }

        # 2. Stats: Text density sampling (first 5 pages)
        # We sample first page and a few others
        stats = {}
        text_samples = []
        for i in range(min(5, doc.page_count)):
            page = doc.load_page(i)
            text = page.get_text()
            text_samples.append(text[:200])  # Sample first 200 chars per page
            stats[f"page_{i}_char_count"] = len(text)

        # 3. Summary Hint
        summary_hint = f"PDF document with {doc.page_count} pages. "
        if structure["metadata"].get("title"):
            summary_hint += f"Title: {structure['metadata']['title']}. "
        if toc:
            summary_hint += f"Contains {len(toc)} sections in ToC."

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="pdf",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet="\n---\n".join(text_samples),
        )
