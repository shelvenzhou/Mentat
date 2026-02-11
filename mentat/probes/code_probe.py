from pathlib import Path
from typing import Dict, Any, List
from mentat.probes.base import BaseProbe, ProbeResult
import tree_sitter_python as tspython
from tree_sitter import Language, Parser


class CodeProbe(BaseProbe):
    def __init__(self):
        self.PY_LANGUAGE = Language(tspython.language())
        self.parser = Parser(self.PY_LANGUAGE)

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".py")  # Start with Python

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "rb") as f:
            content = f.read()

        tree = self.parser.parse(content)

        # 1. Structure: Extract function and class names
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

        structure = {"classes": classes, "functions": functions}

        # 2. Stats: Line count, complexity hint
        lines = content.decode("utf-8").splitlines()
        stats = {
            "line_count": len(lines),
            "class_count": len(classes),
            "function_count": len(functions),
        }

        # 3. Summary Hint
        summary_hint = f"Python source file with {len(classes)} classes and {len(functions)} functions. "
        if classes:
            summary_hint += f"Classes: {', '.join(classes[:3])}. "
        if functions:
            summary_hint += f"Functions: {', '.join(functions[:3])}."

        return ProbeResult(
            doc_id="",
            filename=Path(file_path).name,
            file_type="python",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet=content.decode("utf-8")[:500],
        )
