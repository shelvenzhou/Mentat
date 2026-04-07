"""WikiAdaptor — generate deterministic pages and queue ingest events."""

import logging
from typing import Any, Dict, List

from mentat.core.models import BaseAdaptor
from mentat.wiki.log import WikiLog

logger = logging.getLogger("mentat.wiki")


class WikiAdaptor(BaseAdaptor):
    """Generate deterministic wiki pages when documents are indexed."""

    def __init__(self, generator: "WikiGenerator"):  # noqa: F821
        from mentat.wiki.generator import WikiGenerator  # noqa: F811

        self.generator: WikiGenerator = generator
        self.log = WikiLog(generator.wiki_dir)

    def on_document_indexed(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        try:
            stub = self.generator._storage.get_stub(doc_id)
            if not stub:
                return

            self.generator.generate_page(stub)
            self.log.append_event(
                "ingest",
                filename=stub.get("filename", "unknown"),
                sid=self.generator.short_id(doc_id),
            )
        except Exception:
            logger.warning("Wiki page generation failed for %s", doc_id, exc_info=True)

    def on_search_results(
        self, query: str, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return results
