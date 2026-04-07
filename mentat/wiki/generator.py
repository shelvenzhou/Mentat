"""WikiGenerator — deterministic wiki assets owned by Mentat."""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

from mentat.probes.base import ProbeResult

logger = logging.getLogger("mentat.wiki")


def slugify(text: str) -> str:
    """Convert text to a URL-safe anchor slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text.strip("-")


class WikiGenerator:
    """Generates deterministic wiki files; agents own synthesis files."""

    def __init__(
        self,
        wiki_dir: str,
        storage: Any,
        collections_store: Any,
        file_store: Any,
    ):
        self._wiki_dir = Path(wiki_dir)
        self._pages_dir = self._wiki_dir / "pages"
        self._topics_dir = self._wiki_dir / "topics"
        self._storage = storage
        self._collections_store = collections_store
        self._file_store = file_store

        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._topics_dir.mkdir(parents=True, exist_ok=True)

        self._page_map_path = self._wiki_dir / "_page_map.json"
        self._page_map: Dict[str, str] = self._load_page_map()
        self.ensure_workspace_files()

    @property
    def wiki_dir(self) -> Path:
        return self._wiki_dir

    def ensure_workspace_files(self) -> None:
        """Ensure the shared wiki workspace has the base files agents expect."""
        placeholders = {
            "index.md": (
                "# Mentat Wiki\n\n"
                "_Agent-owned catalog. Run `mentat wiki sync` to generate topics and rewrite this file._\n"
            ),
            "log.md": (
                "# Mentat Wiki Log\n\n"
                "_Append-only timeline of ingest, sync, verify, and lint events._\n"
            ),
        }
        for filename, content in placeholders.items():
            path = self._wiki_dir / filename
            if not path.exists():
                path.write_text(content, "utf-8")

    def _load_page_map(self) -> Dict[str, str]:
        if self._page_map_path.exists():
            try:
                return json.loads(self._page_map_path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_page_map(self) -> None:
        self._page_map_path.write_text(
            json.dumps(self._page_map, indent=2, ensure_ascii=False), "utf-8"
        )

    @staticmethod
    def short_id(doc_id: str) -> str:
        return doc_id[:8]

    def _compact_brief(self, text: str, limit: int = 100) -> str:
        cleaned = re.sub(r"\s+", " ", text or "").strip()
        cleaned = cleaned.replace("|", "\\|")
        return cleaned[:limit]

    def generate_page(self, stub: Dict[str, Any]) -> Path:
        """Generate or update a single deterministic per-document page."""
        self.ensure_workspace_files()

        doc_id: str = stub["id"]
        filename: str = stub.get("filename", "unknown")
        brief_intro: str = stub.get("brief_intro", "")
        source: str = stub.get("source", "")
        probe_json_str: str = stub.get("probe_json", "{}")

        probe = self._parse_probe(probe_json_str, filename)
        sid = self.short_id(doc_id)
        lines: List[str] = []

        title = probe.topic.title if probe and probe.topic.title else filename
        lines.append(f"# {title}")
        lines.append("")

        if brief_intro:
            lines.append(f"> {brief_intro}")
            lines.append("")

        meta_parts = []
        if source:
            meta_parts.append(f"**Source**: `{source}`")
        if probe:
            meta_parts.append(f"**Type**: {probe.file_type}")
        meta_parts.append(f"**ID**: `{sid}`")
        lines.append(" · ".join(meta_parts))
        lines.append("")

        toc_entries = probe.structure.toc if probe else []
        if toc_entries:
            lines.append("## Contents")
            lines.append("")
            for entry in toc_entries:
                indent = "  " * max(0, entry.level - 1)
                slug = slugify(entry.title)
                lines.append(f"{indent}- [{entry.title}](#{slug})")
            lines.append("")
            lines.append("---")
            lines.append("")

        if probe and probe.chunks:
            current_section: Optional[str] = None
            for chunk in probe.chunks:
                section = chunk.section or "Content"
                if section != current_section:
                    current_section = section
                    lines.append(f"## {section}")
                    lines.append("")

                content = chunk.content.strip()
                if content:
                    lines.append(content)
                    lines.append("")
        elif brief_intro:
            lines.append("## Content")
            lines.append("")
            lines.append("_Content is being processed. Check back shortly._")
            lines.append("")

        lines.append("---")
        lines.append(
            f"_Wiki page for **{filename}** · [Full doc](/wiki/pages/{sid})_"
        )

        page_path = self._pages_dir / f"{sid}.md"
        page_path.write_text("\n".join(lines), "utf-8")
        self._page_map[sid] = doc_id
        self._save_page_map()

        logger.debug("Generated wiki page: %s (%s)", sid, filename)
        return page_path

    def generate_memories_page(self) -> Path:
        """Rebuild _memories.md from the 'memory' collection."""
        return self._generate_section_page(
            "memory",
            "Memories",
            "Deterministic collection view of long-term memories.",
            "_memories.md",
        )

    def generate_conversations_page(self) -> Path:
        """Rebuild _conversations.md from the 'chat_history' collection."""
        return self._generate_section_page(
            "chat_history",
            "Conversations",
            "Deterministic collection view of indexed chat history.",
            "_conversations.md",
        )

    def _generate_section_page(
        self, collection_name: str, title: str, description: str, output_name: str
    ) -> Path:
        self.ensure_workspace_files()

        rec = self._collections_store.get(collection_name)
        doc_ids = rec.get("doc_ids", []) if rec else []

        now = time.strftime("%Y-%m-%d %H:%M")
        lines: List[str] = [
            f"# {title}",
            "",
            f"> {description}",
            "",
            f"_Last updated: {now}_ · **{len(doc_ids)}** entries",
            "",
            "[Back to Index](/wiki/)",
            "",
        ]

        if not doc_ids:
            lines.append("_No entries yet._")
        else:
            lines.append("| Document | Brief |")
            lines.append("|----------|-------|")
            for did in doc_ids[-100:]:
                stub = self._storage.get_stub(did)
                if not stub:
                    continue
                sid = self.short_id(did)
                fname = stub.get("filename", "?")
                brief = self._compact_brief(stub.get("brief_intro") or "")
                lines.append(f"| [{fname}](/wiki/pages/{sid}) | {brief} |")

        page_path = self._wiki_dir / output_name
        page_path.write_text("\n".join(lines), "utf-8")
        return page_path

    def rebuild_all(self) -> int:
        """Regenerate deterministic wiki assets from current stubs."""
        self.ensure_workspace_files()

        stubs = self._storage.list_docs()
        count = 0
        for stub in stubs:
            try:
                self.generate_page(stub)
                count += 1
            except Exception:
                logger.warning(
                    "Failed to generate wiki page for %s", stub.get("id"), exc_info=True
                )

        self.generate_memories_page()
        self.generate_conversations_page()
        self._save_page_map()

        logger.info("Wiki deterministic rebuild complete: %d pages", count)
        return count

    def resolve_url(self, url: str) -> Dict[str, Any]:
        """Parse a deterministic page URL into {doc_id, section_path, filename}."""
        fragment: Optional[str] = None

        if url.startswith(("http://", "https://")):
            parsed = urlparse(url)
            path = parsed.path
            fragment = unquote(parsed.fragment) if parsed.fragment else None
        elif "#" in url:
            path, fragment_raw = url.split("#", 1)
            fragment = unquote(fragment_raw)
        else:
            path = url

        normalized_path = path.rstrip("/")
        if normalized_path in ("", "/wiki"):
            return {"error": "index_not_resolvable"}
        if "/wiki/topics/" in normalized_path:
            return {"error": "topic_not_resolvable"}

        page_id = normalized_path.rsplit("/", 1)[-1]
        if page_id.endswith(".md"):
            page_id = page_id[:-3]

        full_doc_id = self._page_map.get(page_id)
        if not full_doc_id:
            return {"error": "page_not_found", "page_id": page_id}

        stub = self._storage.get_stub(full_doc_id)
        filename = stub.get("filename", "unknown") if stub else "unknown"
        section_path: Optional[str] = None
        if fragment and stub:
            section_path = self._resolve_section(stub, fragment)

        return {
            "doc_id": full_doc_id,
            "section_path": section_path,
            "filename": filename,
        }

    def _resolve_section(self, stub: Dict[str, Any], fragment: str) -> Optional[str]:
        probe = self._parse_probe(stub.get("probe_json", "{}"), "")
        if not probe:
            return fragment

        for entry in probe.structure.toc:
            if slugify(entry.title) == fragment:
                return entry.title

        fragment_lower = fragment.replace("-", " ")
        for entry in probe.structure.toc:
            if entry.title.lower() == fragment_lower:
                return entry.title

        return fragment

    @staticmethod
    def _parse_probe(probe_json_str: str, filename: str) -> Optional[ProbeResult]:
        if not probe_json_str or probe_json_str == "{}":
            return None
        try:
            data = json.loads(probe_json_str)
            return ProbeResult.model_validate(data)
        except Exception:
            logger.debug("Could not parse probe_json for %s", filename)
            return None
