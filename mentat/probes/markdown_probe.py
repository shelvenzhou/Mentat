import re
from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import BaseProbe, ProbeResult


class MarkdownProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return (
            filename.lower().endswith((".md", ".markdown"))
            or content_type == "text/markdown"
        )

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 1. Structure: Header hierarchy
        headers = []
        # Match lines starting with #
        header_pattern = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
        for match in header_pattern.finditer(content):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({"level": level, "title": title})

        structure = {"headers": headers}

        # 2. Stats: Link density, code block count, word count
        links = re.findall(r"\[.*?\]\(.*?\)", content)
        code_blocks = re.findall(r"```", content)
        words = content.split()

        stats = {
            "link_count": len(links),
            "code_block_count": len(code_blocks) // 2,
            "word_count": len(words),
            "approx_tokens": len(words) * 1.3,  # Rough estimate
        }

        # 3. Summary Hint
        summary_hint = f"Markdown document with {len(headers)} headers and approx {stats['word_count']} words. "
        if headers:
            summary_hint += (
                f"Main topics include: {', '.join([h['title'] for h in headers[:3]])}."
            )

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="markdown",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet=content[:500],  # First 500 chars
        )
