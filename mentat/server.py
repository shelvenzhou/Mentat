"""HTTP server exposing the Mentat API via FastAPI.

Start with ``mentat serve`` or programmatically::

    from mentat.server import create_app
    import uvicorn
    uvicorn.run(create_app(), host="0.0.0.0", port=7832)
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mentat.core.hub import Mentat, MentatConfig


# ── Request / Response Models ────────────────────────────────────────────


class IndexRequest(BaseModel):
    path: str
    force: bool = False
    summarize: bool = False
    wait: bool = False


class IndexContentRequest(BaseModel):
    content: str
    filename: str
    content_type: str = "text/plain"
    force: bool = False
    summarize: bool = False
    wait: bool = False


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    hybrid: bool = False
    collection: Optional[str] = None
    toc_only: bool = False


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

    def _mentat() -> Mentat:
        return Mentat.get_instance()

    # ── Health ───────────────────────────────────────────────────────

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ── Index ────────────────────────────────────────────────────────

    @app.post("/index")
    async def index_file(req: IndexRequest):
        m = _mentat()
        resolved = str(Path(req.path).resolve())
        if not Path(resolved).is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        doc_id = await m.add(
            resolved, force=req.force, summarize=req.summarize, wait=req.wait
        )
        status = m.get_processing_status(doc_id)
        return {"doc_id": doc_id, "status": status.get("status", "unknown")}

    @app.post("/index-content")
    async def index_content(req: IndexContentRequest):
        m = _mentat()
        doc_id = await m.add_content(
            req.content,
            req.filename,
            content_type=req.content_type,
            force=req.force,
            summarize=req.summarize,
            wait=req.wait,
        )
        status = m.get_processing_status(doc_id)
        return {"doc_id": doc_id, "status": status.get("status", "unknown")}

    # ── Search ───────────────────────────────────────────────────────

    @app.post("/search")
    async def search(req: SearchRequest):
        m = _mentat()
        if req.collection:
            coll = m.collection(req.collection)
            results = await coll.search(req.query, top_k=req.top_k, hybrid=req.hybrid)
        else:
            results = await m.search(
                req.query, top_k=req.top_k, hybrid=req.hybrid, toc_only=req.toc_only
            )
        return {"results": [r.model_dump() for r in results]}

    # ── Status / Inspect ─────────────────────────────────────────────

    @app.get("/status/{doc_id}")
    async def status(doc_id: str):
        return _mentat().get_processing_status(doc_id)

    @app.get("/inspect/{doc_id}")
    async def inspect(doc_id: str):
        result = await _mentat().inspect(doc_id)
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

    # ── Stats ────────────────────────────────────────────────────────

    @app.get("/stats")
    async def stats():
        return _mentat().stats()

    # ── Collections ──────────────────────────────────────────────────

    @app.get("/collections")
    async def list_collections():
        return {"collections": _mentat().list_collections()}

    @app.post("/collections/{name}/add")
    async def collection_add(name: str, req: CollectionAddRequest):
        m = _mentat()
        coll = m.collection(name)
        resolved = str(Path(req.path).resolve())
        if not Path(resolved).is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        doc_id = await coll.add(resolved, force=req.force, summarize=req.summarize)
        return {"doc_id": doc_id, "collection": name}

    @app.post("/collections/{name}/search")
    async def collection_search(name: str, req: SearchRequest):
        m = _mentat()
        coll = m.collection(name)
        results = await coll.search(req.query, top_k=req.top_k, hybrid=req.hybrid)
        return {"results": [r.model_dump() for r in results]}

    return app
