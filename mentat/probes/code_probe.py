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
from mentat.probes._utils import estimate_tokens, should_bypass, extract_preview
from mentat.probes.instruction_templates import (
    CODE_BRIEF_INTRO,
    CODE_INSTRUCTIONS,
)
from tree_sitter import Language, Parser

# ---------------------------------------------------------------------------
# Language registry: each entry maps to a tree-sitter language + extractors
# ---------------------------------------------------------------------------

_LANGUAGES: Dict[str, Dict[str, Any]] = {}


def _register_language(name, module_import, lang_attr, extensions, file_type):
    """Try to register a tree-sitter language; skip silently if unavailable."""
    try:
        mod = __import__(module_import)
        lang_fn = getattr(mod, lang_attr, None) or getattr(mod, "language", None)
        _LANGUAGES[name] = {
            "lang": Language(lang_fn()),
            "extensions": extensions,
            "file_type": file_type,
        }
    except (ImportError, Exception):
        pass


# Always available
import tree_sitter_python as _tspy

_LANGUAGES["python"] = {
    "lang": Language(_tspy.language()),
    "extensions": (".py",),
    "file_type": "python",
}

# Optional: JavaScript
_register_language(
    "javascript",
    "tree_sitter_javascript",
    "language",
    (".js", ".jsx", ".mjs"),
    "javascript",
)

# Optional: TypeScript
try:
    import tree_sitter_typescript as _tsts

    _LANGUAGES["typescript"] = {
        "lang": Language(_tsts.language_typescript()),
        "extensions": (".ts",),
        "file_type": "typescript",
    }
    _LANGUAGES["tsx"] = {
        "lang": Language(_tsts.language_tsx()),
        "extensions": (".tsx",),
        "file_type": "tsx",
    }
except (ImportError, Exception):
    pass


class CodeProbe(BaseProbe):
    """Probe for code files (Python, JavaScript, TypeScript)."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        ext = Path(filename).suffix.lower()
        return any(ext in info["extensions"] for info in _LANGUAGES.values())

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        text = raw_bytes.decode("utf-8", errors="replace")
        approx_tokens = estimate_tokens(text)

        # Determine language
        ext = Path(file_path).suffix.lower()
        lang_name, lang_info = self._get_language(ext)
        if not lang_info:
            # Fallback: treat as plain text
            return self._fallback_result(file_path, text, approx_tokens)

        parser = Parser(lang_info["lang"])
        tree = parser.parse(raw_bytes)
        root = tree.root_node

        # --- Extract imports ---
        imports = self._extract_imports(root, text, lang_name)

        # --- Extract hierarchical definitions ---
        toc_entries, definitions = self._extract_definitions(root, text, lang_name)

        # --- Module-level docstring ---
        abstract = self._extract_module_docstring(root, text, lang_name)

        # --- Stats ---
        lines = text.splitlines()
        class_count = sum(1 for e in toc_entries if e.level == 1 and "class " in (e.annotation or ""))
        func_count = len(definitions) - class_count

        stats: Dict[str, Any] = {
            "line_count": len(lines),
            "class_count": class_count,
            "function_count": func_count,
            "import_count": len(imports),
            "imports": imports[:20],
            "language": lang_info["file_type"],
            "approx_tokens": approx_tokens,
        }

        topic = TopicInfo(
            title=Path(file_path).stem,
            abstract=abstract,
        )

        structure = StructureInfo(
            toc=toc_entries,
            definitions=[d for d in definitions],
        )

        # --- Small-file bypass ---
        if should_bypass(text):
            stats["is_full_content"] = True
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type=lang_info["file_type"],
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[Chunk(content=text, index=0)],
                raw_snippet=text,
            )
            # Generate format-specific instructions
            brief_intro, instructions = self.generate_instructions(result)
            result.brief_intro = brief_intro
            result.instructions = instructions
            return result

        # --- Chunks: one per top-level definition ---
        stats["is_full_content"] = False
        chunks = self._build_chunks(root, text, lang_name)

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type=lang_info["file_type"],
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=text[:500],
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate code-specific instructions."""
        stats = probe_result.stats
        lang = stats['language'].title()  # Python, Javascript, etc.

        # Brief intro
        brief_intro = CODE_BRIEF_INTRO.format(language=lang)

        # Full instructions
        instructions = CODE_INSTRUCTIONS.format(
            language=lang,
            filename=probe_result.filename,
        )

        return brief_intro, instructions

    # ------------------------------------------------------------------
    # Language detection
    # ------------------------------------------------------------------

    def _get_language(self, ext: str) -> Tuple[Optional[str], Optional[Dict]]:
        for name, info in _LANGUAGES.items():
            if ext in info["extensions"]:
                return name, info
        return None, None

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_imports(
        self, root, text: str, lang: str
    ) -> List[str]:
        imports: List[str] = []
        for node in root.children:
            if lang == "python":
                if node.type in ("import_statement", "import_from_statement"):
                    imports.append(text[node.start_byte : node.end_byte].strip())
            elif lang in ("javascript", "typescript", "tsx"):
                if node.type == "import_statement":
                    imports.append(text[node.start_byte : node.end_byte].strip())
                # Handle: export { ... } from '...'
                if node.type == "export_statement":
                    src = node.child_by_field_name("source")
                    if src:
                        imports.append(text[node.start_byte : node.end_byte].strip())
        return imports

    # ------------------------------------------------------------------
    # Definition extraction with hierarchical ToC
    # ------------------------------------------------------------------

    def _extract_definitions(
        self, root, text: str, lang: str
    ) -> Tuple[List[TocEntry], List[str]]:
        """Extract classes, methods, and functions as hierarchical TocEntry list."""
        toc: List[TocEntry] = []
        flat_defs: List[str] = []

        for node in root.children:
            # Handle exported definitions in JS/TS
            actual_node = node
            if node.type == "export_statement" and lang in ("javascript", "typescript", "tsx"):
                for child in node.children:
                    if child.type in (
                        "class_declaration",
                        "function_declaration",
                        "lexical_declaration",
                    ):
                        actual_node = child
                        break

            if actual_node.type in ("class_definition", "class_declaration"):
                self._extract_class(actual_node, text, lang, toc, flat_defs)
            elif actual_node.type in ("function_definition", "function_declaration"):
                self._extract_function(actual_node, text, lang, toc, flat_defs, level=1)
            elif actual_node.type == "decorated_definition" and lang == "python":
                # Python decorated class/function
                for child in actual_node.children:
                    if child.type == "class_definition":
                        self._extract_class(child, text, lang, toc, flat_defs)
                    elif child.type == "function_definition":
                        self._extract_function(child, text, lang, toc, flat_defs, level=1)
            elif actual_node.type == "lexical_declaration" and lang in ("javascript", "typescript", "tsx"):
                # const foo = () => {} or const foo = function() {}
                for decl in actual_node.children:
                    if decl.type == "variable_declarator":
                        val = decl.child_by_field_name("value")
                        if val and val.type in ("arrow_function", "function"):
                            name_node = decl.child_by_field_name("name")
                            name = text[name_node.start_byte : name_node.end_byte] if name_node else "anonymous"
                            sig = self._get_signature_js(val, name, text)
                            line_count = val.end_point[0] - val.start_point[0] + 1
                            flat_defs.append(f"def {name}")
                            toc.append(
                                TocEntry(
                                    level=1,
                                    title=sig,
                                    annotation=f"{line_count} lines",
                                )
                            )

        return toc, flat_defs

    def _extract_class(
        self, node, text: str, lang: str, toc: List[TocEntry], flat_defs: List[str]
    ):
        name_node = node.child_by_field_name("name")
        name = text[name_node.start_byte : name_node.end_byte] if name_node else "?"

        # Base classes / inheritance
        bases = self._get_bases(node, text, lang)
        base_str = f"extends {bases}" if bases else ""

        # Find body node
        body = node.child_by_field_name("body")
        if not body:
            # JS: class_body is a direct child
            for child in node.children:
                if child.type == "class_body":
                    body = child
                    break

        # Count methods
        methods = []
        if body:
            for child in body.children:
                if child.type in ("function_definition", "method_definition"):
                    methods.append(child)
                elif child.type == "decorated_definition" and lang == "python":
                    for sub in child.children:
                        if sub.type == "function_definition":
                            methods.append(sub)

        # Annotation
        parts = []
        if base_str:
            parts.append(base_str)
        parts.append(f"{len(methods)} methods")
        annotation = " | ".join(parts)

        # Docstring as preview
        docstring = self._get_docstring(body, text, lang) if body else None

        flat_defs.append(f"class {name}")
        toc.append(
            TocEntry(level=1, title=f"class {name}", annotation=annotation, preview=docstring)
        )

        # Methods as level-2 entries
        for method_node in methods:
            self._extract_function(method_node, text, lang, toc, flat_defs, level=2)

    def _extract_function(
        self, node, text: str, lang: str, toc: List[TocEntry], flat_defs: List[str], level: int
    ):
        name_node = node.child_by_field_name("name")
        if not name_node:
            # method_definition in JS
            for child in node.children:
                if child.type == "property_identifier":
                    name_node = child
                    break
        name = text[name_node.start_byte : name_node.end_byte] if name_node else "?"

        # Signature
        if lang == "python":
            sig = self._get_signature_py(node, name, text)
        else:
            sig = self._get_signature_js(node, name, text)

        # Line count
        line_count = node.end_point[0] - node.start_point[0] + 1

        # Docstring
        body = node.child_by_field_name("body")
        docstring = self._get_docstring(body, text, lang) if body else None

        flat_defs.append(f"def {name}")
        toc.append(
            TocEntry(
                level=level,
                title=sig,
                annotation=f"{line_count} lines",
                preview=docstring,
            )
        )

    # ------------------------------------------------------------------
    # Signature extraction
    # ------------------------------------------------------------------

    def _get_signature_py(self, node, name: str, text: str) -> str:
        params_node = node.child_by_field_name("parameters")
        params = text[params_node.start_byte : params_node.end_byte] if params_node else "()"
        # Return type
        ret_node = node.child_by_field_name("return_type")
        ret = ""
        if ret_node:
            ret = " -> " + text[ret_node.start_byte : ret_node.end_byte]
        return f"{name}{params}{ret}"

    def _get_signature_js(self, node, name: str, text: str) -> str:
        params_node = node.child_by_field_name("parameters")
        if not params_node:
            # Arrow function or method
            for child in node.children:
                if child.type == "formal_parameters":
                    params_node = child
                    break
        params = text[params_node.start_byte : params_node.end_byte] if params_node else "()"
        return f"{name}{params}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_bases(self, node, text: str, lang: str) -> str:
        if lang == "python":
            # argument_list child contains base classes
            for child in node.children:
                if child.type == "argument_list":
                    return text[child.start_byte + 1 : child.end_byte - 1]
        elif lang in ("javascript", "typescript", "tsx"):
            # class_heritage > extends_clause
            for child in node.children:
                if child.type == "class_heritage":
                    return text[child.start_byte : child.end_byte].replace("extends ", "")
        return ""

    def _get_docstring(self, body_node, text: str, lang: str) -> Optional[str]:
        if not body_node or not body_node.children:
            return None

        if lang == "python":
            first = body_node.children[0]
            if first.type == "expression_statement":
                string_node = first.children[0] if first.children else None
                if string_node and string_node.type == "string":
                    raw = text[string_node.start_byte : string_node.end_byte]
                    # Strip quotes
                    raw = raw.strip("\"'").strip()
                    first_line = raw.split("\n")[0].strip()
                    return first_line[:120] if first_line else None
        elif lang in ("javascript", "typescript", "tsx"):
            # Look for JSDoc comment before the body's parent
            parent = body_node.parent
            if parent and parent.prev_sibling and parent.prev_sibling.type == "comment":
                comment_text = text[parent.prev_sibling.start_byte : parent.prev_sibling.end_byte]
                if comment_text.startswith("/**"):
                    lines = comment_text.split("\n")
                    for line in lines:
                        cleaned = line.strip().lstrip("/* ").rstrip("*/").strip()
                        if cleaned and not cleaned.startswith("@"):
                            return cleaned[:120]
        return None

    def _extract_module_docstring(self, root, text: str, lang: str) -> Optional[str]:
        if lang == "python" and root.children:
            first = root.children[0]
            if first.type == "expression_statement":
                string_node = first.children[0] if first.children else None
                if string_node and string_node.type == "string":
                    raw = text[string_node.start_byte : string_node.end_byte]
                    raw = raw.strip("\"'").strip()
                    return raw[:300]
        return None

    def _build_chunks(self, root, text: str, lang: str) -> List[Chunk]:
        """Build chunks from top-level definitions."""
        chunks: List[Chunk] = []

        for node in root.children:
            actual_node = node
            if node.type == "export_statement" and lang in ("javascript", "typescript", "tsx"):
                for child in node.children:
                    if child.type in (
                        "class_declaration",
                        "function_declaration",
                        "lexical_declaration",
                    ):
                        actual_node = child
                        break

            if actual_node.type in (
                "class_definition",
                "class_declaration",
                "function_definition",
                "function_declaration",
                "decorated_definition",
            ):
                name_node = actual_node.child_by_field_name("name")
                name = text[name_node.start_byte : name_node.end_byte] if name_node else "unknown"
                chunk_text = text[node.start_byte : node.end_byte]
                kind = actual_node.type.replace("_definition", "").replace("_declaration", "")
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        index=len(chunks),
                        section=f"{kind} {name}",
                    )
                )

        if not chunks:
            chunks.append(Chunk(content=text, index=0))

        return chunks

    def _fallback_result(
        self, file_path: str, text: str, approx_tokens: int
    ) -> ProbeResult:
        lines = text.splitlines()
        return ProbeResult(
            filename=Path(file_path).name,
            file_type="code",
            topic=TopicInfo(title=Path(file_path).stem),
            structure=StructureInfo(),
            stats={
                "line_count": len(lines),
                "approx_tokens": approx_tokens,
                "language": "unknown",
                "is_full_content": True,
            },
            chunks=[Chunk(content=text, index=0)],
            raw_snippet=text[:500],
        )
