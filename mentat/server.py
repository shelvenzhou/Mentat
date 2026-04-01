"""HTTP server exposing the Mentat API via FastAPI.

Start with ``mentat serve`` or programmatically::

    from mentat.server import create_app
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=7832)
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401 – Dict used in Pydantic models

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from mentat.core.hub import Mentat, MentatConfig
from mentat import service

logger = logging.getLogger("mentat.server")

# Maximum bytes of request body to log (avoid dumping huge content payloads)
_MAX_BODY_LOG = 1024


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every HTTP request with method, path, body summary, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        path = request.url.path
        method = request.method

        # Read and cache request body for logging
        body_bytes = await request.body()
        body_summary = self._summarise_body(path, body_bytes)

        response: Response = await call_next(request)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d (%.1fms)%s",
            method,
            path,
            response.status_code,
            elapsed_ms,
            f"  {body_summary}" if body_summary else "",
        )
        return response

    @staticmethod
    def _summarise_body(path: str, body_bytes: bytes) -> str:
        """Return a compact, useful summary of the request body."""
        if not body_bytes:
            return ""
        try:
            data = json.loads(body_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return f"[{len(body_bytes)}B binary]"

        # Per-endpoint summaries — show the fields that matter, skip bulky ones
        if path == "/index":
            return f"path={data.get('path')} source={data.get('source', '')}"
        if path == "/index-content":
            content_len = len(data.get("content", ""))
            return (
                f"filename={data.get('filename')} "
                f"source={data.get('source', '')} "
                f"content={content_len} chars"
            )
        if path in ("/search", "/search-grouped"):
            return (
                f"query={data.get('query')!r} "
                f"top_k={data.get('top_k', 5)} "
                f"toc_only={data.get('toc_only', False)}"
            )
        if path == "/read-segment":
            return (
                f"doc_id={data.get('doc_id', '')[:8]}… "
                f"section={data.get('section_path')!r}"
            )
        if path == "/probe":
            if data.get("path"):
                return f"path={data['path']}"
            return f"filename={data.get('filename')} content={len(data.get('content', ''))} chars"

        # Fallback: dump compact JSON, truncated
        compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(compact) > _MAX_BODY_LOG:
            compact = compact[:_MAX_BODY_LOG] + "…"
        return compact


# ── Request / Response Models ────────────────────────────────────────────


class IndexRequest(BaseModel):
    path: str
    force: bool = False
    summarize: bool = False
    wait: bool = False
    source: str = ""
    metadata: Optional[Dict[str, Any]] = None
    collection: Optional[str] = None


class IndexContentRequest(BaseModel):
    content: str
    filename: str
    content_type: str = "text/plain"
    force: bool = False
    summarize: bool = False
    wait: bool = False
    source: str = ""
    metadata: Optional[Dict[str, Any]] = None
    collection: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool = False
    collection: Optional[str] = None
    collections: Optional[List[str]] = None
    toc_only: bool = False
    source: Optional[str] = None
    with_metadata: Optional[bool] = None
    metadata_filter: Optional[Dict[str, Any]] = None


class TrackRequest(BaseModel):
    path: str


class ReadRequest(BaseModel):
    path: str
    sections: Optional[List[str]] = None
    include_content: bool = False


class ProbeRequest(BaseModel):
    path: Optional[str] = None
    content: Optional[str] = None
    filename: Optional[str] = None


class ReadSegmentRequest(BaseModel):
    doc_id: str
    section_path: str
    include_summary: bool = True


class CollectionAddRequest(BaseModel):
    path: str
    force: bool = False
    summarize: bool = False


class CollectionCreateRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None
    watch_paths: Optional[List[str]] = None
    watch_ignore: Optional[List[str]] = None
    auto_add_sources: Optional[List[str]] = None


class CollectionUpdateRequest(BaseModel):
    metadata: Optional[Dict[str, Any]] = None
    watch_paths: Optional[List[str]] = None
    watch_ignore: Optional[List[str]] = None
    auto_add_sources: Optional[List[str]] = None


# ── App Factory ──────────────────────────────────────────────────────────


def create_app(config: Optional[MentatConfig] = None) -> FastAPI:
    """Create a FastAPI application wired to a Mentat instance."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        mentat = Mentat.get_instance(config)
        await mentat.start()
        yield
        await mentat.shutdown()

    app = FastAPI(
        title="Mentat",
        description="Semantic file intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestLoggingMiddleware)

    def _mentat() -> Mentat:
        return Mentat.get_instance()

    # ── Health ───────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Index ────────────────────────────────────────────────────────

    @app.post("/index")
    async def index_file(req: IndexRequest):
        resolved = str(Path(req.path).resolve())
        if not Path(resolved).is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        return await service.index_file(
            req.path,
            force=req.force,
            summarize=req.summarize,
            wait=req.wait,
            source=req.source,
            metadata=req.metadata,
            collection=req.collection,
        )

    @app.post("/index-content")
    async def index_content(req: IndexContentRequest):
        return await service.index_content(
            req.content,
            req.filename,
            content_type=req.content_type,
            force=req.force,
            summarize=req.summarize,
            wait=req.wait,
            source=req.source,
            metadata=req.metadata,
            collection=req.collection,
        )

    # ── Search ───────────────────────────────────────────────────────

    @app.post("/search")
    async def search(req: SearchRequest):
        results = await service.search_docs(
            req.query,
            top_k=req.top_k,
            hybrid=req.hybrid,
            toc_only=req.toc_only,
            source=req.source,
            with_metadata=req.with_metadata,
            collection=req.collection,
            collections=req.collections,
            metadata_filter=req.metadata_filter,
        )
        return {"results": results}

    @app.post("/search-grouped")
    async def search_grouped(req: SearchRequest):
        results = await service.search_grouped(
            req.query,
            top_k=req.top_k,
            hybrid=req.hybrid,
            toc_only=req.toc_only,
            source=req.source,
            with_metadata=req.with_metadata,
            collection=req.collection,
            collections=req.collections,
            metadata_filter=req.metadata_filter,
        )
        return {"results": results}

    # ── Status / Inspect ─────────────────────────────────────────────

    @app.get("/status/{doc_id}")
    async def status(doc_id: str):
        return _mentat().get_processing_status(doc_id)

    @app.get("/doc-meta/{doc_id}")
    async def doc_meta(doc_id: str):
        result = await _mentat().get_doc_meta(doc_id)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return result

    @app.get("/inspect/{doc_id}")
    async def inspect(doc_id: str, sections: Optional[str] = None, full: bool = False):
        section_list = [s.strip() for s in sections.split(",") if s.strip()] if sections else None
        result = await _mentat().inspect(doc_id, sections=section_list, full=full)
        if result is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return result

    # ── Track Access ─────────────────────────────────────────────────

    @app.post("/track")
    async def track(req: TrackRequest):
        return await _mentat().track_access(req.path)

    # ── Read (Structured) ────────────────────────────────────────────

    @app.post("/read")
    async def read_structured(req: ReadRequest):
        return await _mentat().read_structured(
            req.path, sections=req.sections, include_content=req.include_content
        )

    # ── Read Segment (by doc_id + section) ────────────────────────

    @app.post("/read-segment")
    async def read_segment(req: ReadSegmentRequest):
        result = await _mentat().read_segment(
            req.doc_id, req.section_path, include_summary=req.include_summary
        )
        if result.get("error") == "document_not_found":
            raise HTTPException(
                status_code=404, detail=f"Document not found: {req.doc_id}"
            )
        return result

    # ── Probe ────────────────────────────────────────────────────────

    @app.post("/probe")
    async def probe(req: ProbeRequest):
        from mentat.probes import run_probe

        if req.path:
            resolved = str(Path(req.path).resolve())
            if not Path(resolved).is_file():
                raise HTTPException(
                    status_code=404, detail=f"File not found: {req.path}"
                )
            result = run_probe(resolved)
            return result.model_dump()
        elif req.content and req.filename:
            # Write content to temp file for probing
            import tempfile

            suffix = Path(req.filename).suffix or ".txt"
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False, encoding="utf-8"
            ) as f:
                f.write(req.content)
                tmp_path = f.name
            try:
                result = run_probe(tmp_path)
                result.filename = req.filename
                return result.model_dump()
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'path' or both 'content' and 'filename'",
            )

    # ── Skill ──────────────────────────────────────────────────────────

    @app.get("/skill")
    async def skill():
        from mentat.skill import export_skill

        return export_skill()

    # ── Section Heat ─────────────────────────────────────────────────

    @app.get("/section-heat")
    async def section_heat(doc_id: Optional[str] = None, limit: int = 20):
        return {"sections": _mentat().get_section_heat(doc_id=doc_id, limit=limit)}

    # ── Stats ────────────────────────────────────────────────────────

    @app.get("/stats")
    async def stats():
        return service.get_stats()

    # ── Collections ──────────────────────────────────────────────────

    @app.get("/collections")
    async def list_collections():
        m = _mentat()
        names = m.collections_store.list_collections()
        result = []
        for name in names:
            rec = m.collections_store.get(name)
            result.append({
                "name": name,
                "doc_count": len(rec["doc_ids"]) if rec else 0,
                "metadata": rec.get("metadata", {}) if rec else {},
                "created_at": rec.get("created_at", "") if rec else "",
            })
        return {"collections": result}

    @app.post("/collections/gc")
    async def collections_gc():
        m = _mentat()
        deleted = m.collections_store.gc()
        return {"deleted": deleted}

    @app.post("/collections/{name}")
    async def create_collection(name: str, req: CollectionCreateRequest):
        m = _mentat()
        rec = m.collections_store.create(
            name,
            metadata=req.metadata,
            watch_paths=req.watch_paths,
            watch_ignore=req.watch_ignore,
            auto_add_sources=req.auto_add_sources,
        )
        await m.watcher.sync()
        return {"name": name, "doc_count": len(rec["doc_ids"]), **rec}

    @app.put("/collections/{name}")
    async def update_collection(name: str, req: CollectionUpdateRequest):
        m = _mentat()
        if m.collections_store.get(name) is None:
            raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
        rec = m.collections_store.create(
            name,
            metadata=req.metadata,
            watch_paths=req.watch_paths,
            watch_ignore=req.watch_ignore,
            auto_add_sources=req.auto_add_sources,
        )
        await m.watcher.sync()
        return {"name": name, "doc_count": len(rec["doc_ids"]), **rec}

    @app.get("/collections/{name}")
    async def get_collection(name: str):
        m = _mentat()
        rec = m.collections_store.get(name)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
        return {"name": name, "doc_count": len(rec["doc_ids"]), **rec}

    @app.delete("/collections/{name}")
    async def delete_collection(name: str):
        m = _mentat()
        if not m.collections_store.delete_collection(name):
            raise HTTPException(status_code=404, detail=f"Collection not found: {name}")
        await m.watcher.sync()
        return {"deleted": name}

    @app.post("/collections/{name}/add")
    async def collection_add(name: str, req: CollectionAddRequest):
        m = _mentat()
        coll = m.collection(name)
        resolved = str(Path(req.path).resolve())
        if not Path(resolved).is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        doc_id = await coll.add(resolved, force=req.force, summarize=req.summarize)
        return {"doc_id": doc_id, "collection": name}

    @app.delete("/collections/{name}/docs/{doc_id}")
    async def collection_remove_doc(name: str, doc_id: str):
        m = _mentat()
        m.collections_store.remove_doc(name, doc_id)
        return {"removed": doc_id, "collection": name}

    @app.post("/collections/{name}/search")
    async def collection_search(name: str, req: SearchRequest):
        m = _mentat()
        coll = m.collection(name)
        results = await coll.search(req.query, top_k=req.top_k, hybrid=req.hybrid)
        return {"results": [r.model_dump() for r in results]}

    return app
