import litellm
import json
from typing import Dict, Any, Tuple
from mentat.probes.base import ProbeResult


class Librarian:
    """Layer 3: Instruction generation layer.

    The Librarian receives ONLY probe results (not the raw file) and generates:
    1. Brief Intro: what this file is about
    2. Actionable Instructions: how a downstream model should use this file
    """

    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def _build_prompt(self, probe_result: ProbeResult) -> str:
        # Build topic summary
        topic_parts = []
        if probe_result.topic.title:
            topic_parts.append(f"Title: {probe_result.topic.title}")
        if probe_result.topic.abstract:
            topic_parts.append(f"Abstract: {probe_result.topic.abstract}")
        if probe_result.topic.first_paragraph:
            topic_parts.append(
                f"First paragraph: {probe_result.topic.first_paragraph[:300]}"
            )
        topic_summary = (
            "\n".join(topic_parts) if topic_parts else "No topic info available."
        )

        # Build structure summary
        structure = probe_result.structure
        struct_parts = []
        if structure.toc:
            toc_str = "\n".join(
                f"  {'  ' * (e.level - 1)}- {e.title}"
                + (f" (p.{e.page})" if e.page else "")
                for e in structure.toc[:20]
            )
            struct_parts.append(f"Table of Contents:\n{toc_str}")
        if structure.captions:
            caps = ", ".join(c.text[:60] for c in structure.captions[:10])
            struct_parts.append(f"Captions: {caps}")
        if structure.columns:
            struct_parts.append(f"Columns: {', '.join(structure.columns)}")
        if structure.schema_tree:
            struct_parts.append(
                f"Schema: {json.dumps(structure.schema_tree, default=str)[:500]}"
            )
        if structure.definitions:
            struct_parts.append(f"Definitions: {', '.join(structure.definitions[:20])}")
        structure_summary = (
            "\n".join(struct_parts) if struct_parts else "No structure info."
        )

        # Build stats summary
        stats_summary = json.dumps(probe_result.stats, indent=2, default=str)[:500]

        # Chunk info
        chunk_info = f"{len(probe_result.chunks)} chunks available."

        prompt = f"""You are 'The Librarian' for the Mentat system.
Your job is to provide a 'Reading Guide' for a model that will use this file.
Since we want to save tokens, the model should NOT read the full file if it doesn't have to.

[File Info]
Filename: {probe_result.filename}
Type: {probe_result.file_type}

[Topic]
{topic_summary}

[Structure]
{structure_summary}

[Statistics]
{stats_summary}

[Chunks]
{chunk_info}

Based on the above, please output:
1. "brief_intro": A 1-2 sentence high-level summary of what this file is.
2. "instructions": Strategic guidance on how to use this file.
   (e.g., "This is a large CSV with 100K rows, use pandas to filter Column X",
    "See Section 3 for the core algorithm", "Run head/tail to preview data")

Output in JSON format:
{{
    "brief_intro": "...",
    "instructions": "..."
}}
"""
        return prompt

    async def generate_guide(self, probe_result: ProbeResult) -> Tuple[str, str, int]:
        """Generate a reading guide from probe results.

        Returns: (brief_intro, instructions, token_usage)
        """
        prompt = self._build_prompt(probe_result)

        response = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        token_usage = response.usage.total_tokens

        return data.get("brief_intro", ""), data.get("instructions", ""), token_usage
