import trafilatura
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import BaseProbe, ProbeResult


class WebProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".html", ".htm")) or content_type == "text/html"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # 1. Structure: Extract main content and metadata using trafilatura
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

        structure = {
            "title": data.get("title"),
            "author": data.get("author"),
            "hostname": data.get("hostname"),
            "date": data.get("date"),
            "categories": data.get("categories"),
            "tags": data.get("tags"),
        }

        # 2. Stats: Content length, readability
        content_text = data.get("text", "")
        stats = {
            "content_length": len(content_text),
            "word_count": len(content_text.split()),
            "approx_tokens": len(content_text.split()) * 1.3,
        }

        # 3. Summary Hint
        summary_hint = f"Web page from {structure['hostname'] or 'unknown source'}. "
        if structure["title"]:
            summary_hint += f"Title: {structure['title']}. "
        summary_hint += f"Main content length: {stats['word_count']} words."

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="web",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet=content_text[:500],
        )
