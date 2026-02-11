import litellm
import json
from typing import Dict, Any, Tuple
from mentat.probes.base import ProbeResult


class Librarian:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def _build_prompt(self, probe_result: ProbeResult) -> str:
        prompt = f"""
        You are 'The Librarian' for the Mentat system.
        Your job is to provide a 'Reading Guide' for a model that will use this file.
        Since we want to save tokens, the model should NOT read the full file if it doesn't have to.

        [File Info]
        Filename: {probe_result.filename}
        Type: {probe_result.file_type}
        
        [Statistical/Structural Probe Results]
        Structure: {json.dumps(probe_result.structure, indent=2)}
        Stats: {json.dumps(probe_result.stats, indent=2)}
        
        [Data Sample/Summary Hint]
        {probe_result.summary_hint}
        Snippet: {probe_result.raw_snippet}

        Based on the above, please output:
        1. "Brief Intro": A 1-2 sentence high-level summary of what this file is.
        2. "Actionable Instructions": Strategic guidance on how to use this file. 
           (e.g., "This is a large CSV, use pandas to filter Column X", "See Page 42 for the core algorithm").

        Output in JSON format:
        {{
            "brief_intro": "...",
            "instructions": "..."
        }}
        """
        return prompt

    async def generate_guide(self, probe_result: ProbeResult) -> Tuple[str, str]:
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
