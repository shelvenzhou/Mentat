import json
from pathlib import Path
from typing import Dict, Any, List, Union
from mentat.probes.base import BaseProbe, ProbeResult


class JSONProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".json") or content_type == "application/json"

    def _infer_schema(self, data: Any, depth: int = 0, max_depth: int = 5) -> Any:
        if depth > max_depth:
            return "..."

        if isinstance(data, dict):
            return {
                k: self._infer_schema(v, depth + 1, max_depth)
                for k, v in data.items()
                if depth < 2 or k in ["id", "type", "name"]
            }
        elif isinstance(data, list):
            if not data:
                return []
            return [self._infer_schema(data[0], depth + 1, max_depth)]
        else:
            return type(data).__name__

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 1. Structure: Schema tree
        schema_tree = self._infer_schema(data)
        structure = {"schema_tree": schema_tree}

        # 2. Stats: Value distribution (if list of dicts)
        stats = {}
        if isinstance(data, list):
            stats["item_count"] = len(data)
            if len(data) > 0 and isinstance(data[0], dict):
                stats["keys"] = list(data[0].keys())
        elif isinstance(data, dict):
            stats["top_level_keys"] = list(data.keys())
            stats["key_count"] = len(data)

        # 3. Summary Hint
        summary_hint = f"JSON file with {stats.get('item_count', stats.get('key_count', 0))} elements. "
        summary_hint += f"Structure follows: {list(schema_tree.keys())[:5] if isinstance(schema_tree, dict) else 'list'}."

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="json",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet=json.dumps(data, indent=2)[:500],
        )
