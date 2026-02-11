import json
import trafilatura
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    Chunk,
)


class WebProbe(BaseProbe):
    """Probe for HTML/Web pages."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".html", ".htm")) or content_type == "text/html"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Use trafilatura for main content extraction (popular library per design)
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

        # --- Topic ---
        topic = TopicInfo(
            title=data.get("title"),
            first_paragraph=content_text[:500] if content_text else None,
        )

        # --- Structure ---
        structure = (
            StructureInfo()
        )  # Web pages don't have a strong structural hierarchy

        # --- Stats ---
        stats = {
            "hostname": data.get("hostname"),
            "author": data.get("author"),
            "date": data.get("date"),
            "categories": data.get("categories"),
            "tags": data.get("tags"),
            "content_length": len(content_text),
            "word_count": len(content_text.split()),
            "approx_tokens": int(len(content_text.split()) * 1.3),
        }

        # --- Chunks: main content ---
        chunks = []
        if content_text:
            # Split by paragraphs (double newline)
            paragraphs = [p.strip() for p in content_text.split("\n\n") if p.strip()]
            for i, para in enumerate(paragraphs):
                chunks.append(Chunk(content=para, index=i))

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="web",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=content_text[:500],
        )
