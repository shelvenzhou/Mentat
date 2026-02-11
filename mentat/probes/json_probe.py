import json as json_module
from pathlib import Path
from typing import Dict, Any, List, Union
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    Chunk,
)


class JSONProbe(BaseProbe):
    """Probe for JSON files."""

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
            data = json_module.load(f)

        # --- Structure: Schema tree ---
        schema_tree = self._infer_schema(data)

        # --- Stats ---
        stats = {}
        if isinstance(data, list):
            stats["item_count"] = len(data)
            if len(data) > 0 and isinstance(data[0], dict):
                stats["keys"] = list(data[0].keys())
        elif isinstance(data, dict):
            stats["top_level_keys"] = list(data.keys())
            stats["key_count"] = len(data)

        # --- Topic ---
        topic = TopicInfo(title=Path(file_path).stem)

        structure = StructureInfo(schema_tree=schema_tree)

        # --- Chunks: truncated sample ---
        raw = json_module.dumps(data, indent=2, ensure_ascii=False)
        chunks = [
            Chunk(
                content=raw[:2000],
                index=0,
                section="root",
                metadata={"truncated": len(raw) > 2000, "total_size": len(raw)},
            )
        ]

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="json",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=raw[:500],
        )
