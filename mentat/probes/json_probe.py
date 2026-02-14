import json as json_module
from pathlib import Path
from typing import Any, Dict, List, Optional
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes._utils import (
    estimate_tokens,
    should_bypass,
    truncate_string,
    extract_preview,
    merge_small_chunks,
)


class JSONProbe(BaseProbe):
    """Probe for JSON files."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".json") or content_type == "application/json"

    def _infer_schema(
        self, data: Any, depth: int = 0, max_depth: int = 3
    ) -> Any:
        if depth >= max_depth:
            if isinstance(data, dict):
                return f"{{... ({len(data)} keys)}}"
            elif isinstance(data, list):
                return f"[... ({len(data)} items)]"
            return type(data).__name__

        if isinstance(data, dict):
            return {
                k: self._infer_schema(v, depth + 1, max_depth)
                for k, v in data.items()
            }
        elif isinstance(data, list):
            if not data:
                return "[]"
            # Homogeneous array of dicts: show first as schema example
            if all(isinstance(item, dict) for item in data):
                schema = self._infer_schema(data[0], depth + 1, max_depth)
                return {"__array_of__": schema, "__count__": len(data)}
            # Homogeneous primitive array
            if all(type(item) == type(data[0]) for item in data):
                return f"[{type(data[0]).__name__} x {len(data)}]"
            return [self._infer_schema(data[0], depth + 1, max_depth)]
        elif isinstance(data, str):
            if len(data) > 40:
                return truncate_string(data)
            return f"str({data})"
        elif isinstance(data, bool):
            return "bool"
        elif isinstance(data, int):
            return "int"
        elif isinstance(data, float):
            return "float"
        elif data is None:
            return "null"
        else:
            return type(data).__name__

    def _compute_depth(self, data: Any, depth: int = 0) -> int:
        if isinstance(data, dict):
            if not data:
                return depth
            return max(self._compute_depth(v, depth + 1) for v in data.values())
        elif isinstance(data, list):
            if not data:
                return depth
            return self._compute_depth(data[0], depth + 1)
        return depth

    def _count_keys(self, data: Any) -> int:
        if isinstance(data, dict):
            count = len(data)
            for v in data.values():
                count += self._count_keys(v)
            return count
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            return self._count_keys(data[0])
        return 0

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()

        data = json_module.loads(raw)
        approx_tokens = estimate_tokens(raw)

        # --- Schema ---
        schema_tree = self._infer_schema(data)

        # --- Stats ---
        stats: Dict[str, Any] = {
            "approx_tokens": approx_tokens,
            "max_depth": self._compute_depth(data),
            "total_keys_count": self._count_keys(data),
        }

        if isinstance(data, list):
            stats["item_count"] = len(data)
            if data and isinstance(data[0], dict):
                stats["keys"] = list(data[0].keys())
        elif isinstance(data, dict):
            stats["top_level_keys"] = list(data.keys())
            stats["key_count"] = len(data)

        # --- Topic ---
        topic = TopicInfo(title=Path(file_path).stem)

        # --- ToC for dict-at-root: one entry per top-level key ---
        toc_entries: List[TocEntry] = []
        if isinstance(data, dict):
            for key, val in data.items():
                annotation = self._describe_value(val)
                preview = self._value_preview(val)
                toc_entries.append(
                    TocEntry(level=1, title=key, annotation=annotation, preview=preview)
                )
        elif isinstance(data, list) and data:
            annotation = f"{len(data)} items"
            if isinstance(data[0], dict):
                annotation += f", keys: {', '.join(list(data[0].keys())[:5])}"
            toc_entries.append(
                TocEntry(level=1, title="root[]", annotation=annotation)
            )

        structure = StructureInfo(schema_tree=schema_tree, toc=toc_entries)

        # --- Small-file bypass ---
        if should_bypass(raw):
            stats["is_full_content"] = True
            return ProbeResult(
                filename=Path(file_path).name,
                file_type="json",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[Chunk(content=raw, index=0, section="root")],
                raw_snippet=raw,
            )

        # --- Chunking ---
        stats["is_full_content"] = False
        chunks = self._build_chunks(data, raw)

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="json",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=raw[:500],
        )

    def _describe_value(self, val: Any) -> str:
        if isinstance(val, dict):
            return f"dict, {len(val)} keys"
        elif isinstance(val, list):
            if not val:
                return "list, empty"
            if isinstance(val[0], dict):
                return f"list[dict], {len(val)} items"
            return f"list[{type(val[0]).__name__}], {len(val)} items"
        elif isinstance(val, str):
            if len(val) > 40:
                return f"str, {len(val)} chars"
            return f"str"
        else:
            return type(val).__name__

    def _value_preview(self, val: Any) -> Optional[str]:
        if isinstance(val, str):
            return truncate_string(val, head=30, tail=15) if len(val) > 50 else val
        elif isinstance(val, (int, float, bool)):
            return str(val)
        elif isinstance(val, list) and val and not isinstance(val[0], (dict, list)):
            sample = ", ".join(str(v) for v in val[:3])
            if len(val) > 3:
                sample += ", ..."
            return f"[{sample}]"
        return None

    def _build_chunks(self, data: Any, raw: str) -> List[Chunk]:
        """Build semantic chunks - one chunk per JSON key/semantic unit.

        Pure semantic splitting: each top-level key becomes a chunk with its
        full serialized value. No artificial windowing or size thresholds.
        The section field maps directly to ToC entries.
        """
        chunks: List[Chunk] = []

        if isinstance(data, dict):
            # One chunk per top-level key - pure semantic splitting
            for key, val in data.items():
                serialized = json_module.dumps(
                    {key: val}, indent=2, ensure_ascii=False, default=str
                )
                chunks.append(
                    Chunk(
                        content=serialized,
                        index=len(chunks),
                        section=key,  # Maps directly to ToC entry
                    )
                )

        elif isinstance(data, list):
            # First item as sample (semantic unit for array structure)
            if data:
                sample = json_module.dumps(
                    data[0], indent=2, ensure_ascii=False, default=str
                )
                section = f"item[0] of {len(data)}"
                chunks.append(
                    Chunk(
                        content=sample,
                        index=0,
                        section=section,
                        metadata={"total_items": len(data)},
                    )
                )
        else:
            # Primitive root value (single semantic unit)
            chunks.append(Chunk(content=raw, index=0, section="root"))

        if not chunks:
            return [Chunk(content=raw[:2000], index=0, section="root")]

        # Apply merge_small_chunks to handle any tiny chunks
        # This is the proper place for chunk size optimization
        return merge_small_chunks(chunks)
