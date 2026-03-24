import configparser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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
    safe_read_text,
    truncate_string,
)
from mentat.librarian.instruction_templates import (
    CONFIG_BRIEF_INTRO,
    CONFIG_INSTRUCTIONS,
    CONFIG_PARSER_YAML,
    CONFIG_PARSER_TOML,
    CONFIG_PARSER_INI,
)


class ConfigProbe(BaseProbe):
    """Probe for config files (YAML, TOML, INI)."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(
            (".yaml", ".yml", ".toml", ".ini", ".conf", ".cfg")
        )

    def run(self, file_path: str) -> ProbeResult:
        content = safe_read_text(file_path)
        ext = Path(file_path).suffix.lower()
        approx_tokens = estimate_tokens(content)

        # --- Parse config ---
        data, config_format = self._parse_config(file_path, ext)

        # --- Build key hierarchy ToC ---
        toc_entries: List[TocEntry] = []
        if data is not None:
            self._walk_keys(data, toc_entries, depth=0, max_depth=3)

        # --- Schema tree (reuse JSON-like structure) ---
        schema_tree = self._infer_schema(data) if data is not None else None

        # --- Stats ---
        key_count = self._count_keys(data) if data is not None else 0
        max_depth = self._compute_depth(data) if data is not None else 0
        top_level_keys = list(data.keys()) if isinstance(data, dict) else []

        stats: Dict[str, Any] = {
            "format": config_format,
            "approx_tokens": approx_tokens,
            "key_count": key_count,
            "max_depth": max_depth,
            "top_level_keys": top_level_keys,
        }

        # --- Topic ---
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=(
                f"{config_format.upper()} config file with {key_count} keys "
                f"across {len(top_level_keys)} top-level sections: "
                f"{', '.join(top_level_keys[:5])}"
                + ("..." if len(top_level_keys) > 5 else "")
            ),
        )

        structure = StructureInfo(
            toc=toc_entries,
            schema_tree=schema_tree,
        )

        # --- Most configs are small, return full content ---
        if should_bypass(content):
            stats["is_full_content"] = True
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type="config",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[Chunk(content=content, index=0)],
                raw_snippet=content,
            )
            brief_intro, instructions = self.generate_instructions(result)
            result.brief_intro = brief_intro
            result.instructions = instructions
            return result

        stats["is_full_content"] = False
        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="config",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=[Chunk(content=content[:2000], index=0, section="root")],
            raw_snippet=content[:500],
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate Config-specific instructions."""
        stats = probe_result.stats
        config_format = stats.get('format', 'unknown')

        # Parser info mapping
        parser_map = {
            'yaml': CONFIG_PARSER_YAML,
            'toml': CONFIG_PARSER_TOML,
            'ini': CONFIG_PARSER_INI,
        }
        parser_info = parser_map.get(config_format, 'unknown parser')

        # Brief intro
        brief_intro = CONFIG_BRIEF_INTRO.format(
            format=config_format.upper(),
            parser=parser_info,
        )

        # Full instructions
        instructions = CONFIG_INSTRUCTIONS.format(
            format=config_format.upper(),
            parser_info=parser_info,
        )

        return brief_intro, instructions

    def _parse_config(self, file_path: str, ext: str):
        if ext in (".yaml", ".yml"):
            import yaml

            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {"root": data}, "yaml"

        elif ext == ".toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            with open(file_path, "rb") as f:
                return tomllib.load(f), "toml"

        elif ext in (".ini", ".conf", ".cfg"):
            cfg = configparser.ConfigParser()
            cfg.read(file_path, encoding="utf-8")
            data = {}
            for section in cfg.sections():
                data[section] = dict(cfg[section])
            if cfg.defaults():
                data["DEFAULT"] = dict(cfg.defaults())
            return data, "ini"

        return None, "unknown"

    def _walk_keys(
        self,
        data: Any,
        entries: List[TocEntry],
        depth: int,
        max_depth: int,
        prefix: str = "",
    ):
        if depth >= max_depth or not isinstance(data, dict):
            return

        for key, val in data.items():
            annotation = self._describe_value(val)
            preview = self._value_preview(val)
            entries.append(
                TocEntry(
                    level=depth + 1,
                    title=key,
                    annotation=annotation,
                    preview=preview,
                )
            )
            if isinstance(val, dict):
                self._walk_keys(val, entries, depth + 1, max_depth)

    def _describe_value(self, val: Any) -> str:
        if isinstance(val, dict):
            return f"dict, {len(val)} keys"
        elif isinstance(val, list):
            if not val:
                return "list, empty"
            return f"list[{type(val[0]).__name__}], {len(val)} items"
        elif isinstance(val, str):
            return "str"
        elif isinstance(val, bool):
            return "bool"
        elif isinstance(val, int):
            return "int"
        elif isinstance(val, float):
            return "float"
        elif val is None:
            return "null"
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

    def _infer_schema(self, data: Any, depth: int = 0, max_depth: int = 3) -> Any:
        if depth >= max_depth:
            if isinstance(data, dict):
                return f"{{... ({len(data)} keys)}}"
            elif isinstance(data, list):
                return f"[... ({len(data)} items)]"
            return type(data).__name__

        if isinstance(data, dict):
            return {k: self._infer_schema(v, depth + 1, max_depth) for k, v in data.items()}
        elif isinstance(data, list):
            if not data:
                return "[]"
            return [self._infer_schema(data[0], depth + 1, max_depth)]
        elif isinstance(data, str):
            return "str"
        elif data is None:
            return "null"
        return type(data).__name__

    def _count_keys(self, data: Any) -> int:
        if isinstance(data, dict):
            count = len(data)
            for v in data.values():
                count += self._count_keys(v)
            return count
        return 0

    def _compute_depth(self, data: Any, depth: int = 0) -> int:
        if isinstance(data, dict) and data:
            return max(self._compute_depth(v, depth + 1) for v in data.values())
        return depth
