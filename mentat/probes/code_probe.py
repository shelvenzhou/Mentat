from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    Chunk,
)
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


class CodeProbe(BaseProbe):
    """Probe for code files (Python)."""

    def __init__(self):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".py")  # Start with Python

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "rb") as f:
            content = f.read()

        tree = self.parser.parse(content)

        # Extract function and class names
        query = self.PY_LANGUAGE.query(
            """
            (class_definition name: (identifier) @class_name)
            (function_definition name: (identifier) @func_name)
        """
        )

        from tree_sitter import QueryCursor

        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)

        classes = [n.text.decode("utf-8") for n in captures.get("class_name", [])]
        functions = [n.text.decode("utf-8") for n in captures.get("func_name", [])]

        # --- Structure: definitions list ---
        all_defs = [f"class {c}" for c in classes] + [f"def {fn}" for fn in functions]

        structure = StructureInfo(definitions=all_defs)

        # --- Topic ---
        text = content.decode("utf-8")
        lines = text.splitlines()

        # Extract docstring as abstract if present
        abstract = None
        if lines and '"""' in text:
            import re

            docstring_match = re.search(r'"""(.*?)"""', text, re.DOTALL)
            if docstring_match:
                abstract = docstring_match.group(1).strip()[:300]

        topic = TopicInfo(
            title=Path(file_path).stem,
            abstract=abstract,
        )

        # --- Stats ---
        stats = {
            "line_count": len(lines),
            "class_count": len(classes),
            "function_count": len(functions),
        }

        # --- Chunks: one chunk per class/function definition ---
        chunks = self._build_chunks(tree, text, classes, functions)

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="python",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=text[:500],
        )

    def _build_chunks(
        self, tree, text: str, classes: list, functions: list
    ) -> List[Chunk]:
        """Build chunks from top-level class and function definitions."""
        chunks = []
        for node in tree.root_node.children:
            if node.type in ("class_definition", "function_definition"):
                name_node = node.child_by_field_name("name")
                name = name_node.text.decode("utf-8") if name_node else "unknown"
                chunk_text = text[node.start_byte : node.end_byte]
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=len(chunks),
                        section=f"{node.type.replace('_definition', '')} {name}",
                    )
                )

        # If no top-level defs, just chunk the whole file
        if not chunks:
            chunks.append(Chunk(content=text, index=0))

        return chunks
