#!/usr/bin/env python3
"""Benchmark: Mentat vs LanceDB RAG vs Naive (whole-file).

Uses the "Attention Is All You Need" paper as the test resource.

Compare targets:
  mentat      — probe + ToC + chunk summaries + targeted retrieval
  lancedb     — standard RAG (embed fixed chunks, vector search)
  naive       — read the whole file, send everything as context

Three experiments:
  1. Trivial RAG     — factual questions answerable by vector search
  2. Summary         — generate a comprehensive paper summary
  3. Agentic Scene   — multi-part questions requiring two-step Q&A

Measures: token usage (embedding + LLM) and latency.

Usage:
    uv run python benchmarks/benchmark.py
    uv run python benchmarks/benchmark.py --experiments trivial_rag summary agentic
    uv run python benchmarks/benchmark.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import litellm
import litellm.integrations.custom_logger
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
litellm.set_verbose = False


# ── Configuration ───────────────────────────────────────────────────────


@dataclass
class BenchmarkConfig:
    paper_path: str = "benchmarks/1706.03762v7.pdf"
    output_file: str = "benchmarks/results.json"

    embedding_model: str = ""
    embedding_dims: int = 3072
    chat_model: str = ""

    api_key: str = ""
    api_base: str = ""

    lancedb_db_path: str = "./.benchmark_lancedb"
    lancedb_chunk_size: int = 1000
    lancedb_chunk_overlap: int = 200

    mentat_db_path: str = "./.benchmark_mentat"
    mentat_storage_dir: str = "./.benchmark_mentat_files"

    top_k: int = 5
    verbose: bool = False

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "")
        self.api_base = self.api_base or os.getenv("OPENAI_API_BASE", "")
        self.chat_model = self.chat_model or os.getenv(
            "BENCHMARK_CHAT_MODEL", "openai/gpt-4o-mini"
        )
        self.embedding_model = self.embedding_model or os.getenv(
            "MENTAT_EMBEDDING_MODEL", "openai/text-embedding-3-large"
        )


# ── Data Models ─────────────────────────────────────────────────────────


@dataclass
class TokenUsage:
    embedding_tokens: int = 0
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    wall_time: float = 0.0

    @property
    def total_llm(self) -> int:
        return self.llm_prompt_tokens + self.llm_completion_tokens


@dataclass
class QuestionResult:
    question_id: str
    query: str
    answer: str = ""
    retrieval: TokenUsage = field(default_factory=TokenUsage)
    answer_gen: TokenUsage = field(default_factory=TokenUsage)
    context_tokens_est: int = 0


@dataclass
class ExperimentResult:
    experiment: str
    system: str
    indexing: TokenUsage = field(default_factory=TokenUsage)
    questions: List[QuestionResult] = field(default_factory=list)

    @property
    def total_embedding_tokens(self) -> int:
        return self.indexing.embedding_tokens + sum(
            q.retrieval.embedding_tokens for q in self.questions
        )

    @property
    def total_llm_tokens(self) -> int:
        idx = self.indexing.total_llm
        q_total = sum(
            q.retrieval.total_llm + q.answer_gen.total_llm
            for q in self.questions
        )
        return idx + q_total

    @property
    def total_time(self) -> float:
        return self.indexing.wall_time + sum(
            q.retrieval.wall_time + q.answer_gen.wall_time
            for q in self.questions
        )


@dataclass
class BenchmarkQuestion:
    id: str
    experiment: str
    query: str
    expected_answer: str
    verification_points: List[str]


# ── Token Tracking ──────────────────────────────────────────────────────


class TokenCallback(litellm.integrations.custom_logger.CustomLogger):
    """Intercepts all litellm calls to track token usage.

    NOTE: litellm fires async callbacks AFTER the response is yielded,
    so callers must ``await asyncio.sleep(0)`` before changing the target.
    """

    def __init__(self):
        super().__init__()
        self.target: Optional[TokenUsage] = None

    def set_target(self, target: Optional[TokenUsage]):
        self.target = target

    async def set_target_async(self, target: Optional[TokenUsage]):
        """Switch target, giving pending callbacks a chance to fire first."""
        await asyncio.sleep(0.05)
        self.target = target

    def _record(self, response_obj, kwargs):
        if self.target is None:
            return
        usage = getattr(response_obj, "usage", None)
        if not usage:
            return
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        is_embed = kwargs.get("call_type", "") in ("embedding", "aembedding")
        if is_embed:
            self.target.embedding_tokens += prompt
        else:
            self.target.llm_prompt_tokens += prompt
            self.target.llm_completion_tokens += completion

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(response_obj, kwargs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._record(response_obj, kwargs)


_cb = TokenCallback()
litellm.callbacks = [_cb]


# ── Questions ───────────────────────────────────────────────────────────

TRIVIAL_RAG_QUESTIONS = [
    BenchmarkQuestion(
        id="T1",
        experiment="trivial_rag",
        query=(
            "What BLEU score did the Transformer (big) model achieve on the "
            "WMT 2014 English-to-German translation task?"
        ),
        expected_answer=(
            "28.4 BLEU, improving over existing best results including "
            "ensembles by over 2 BLEU"
        ),
        verification_points=["28.4", "English-to-German"],
    ),
    BenchmarkQuestion(
        id="T2",
        experiment="trivial_rag",
        query=(
            "How many attention heads does the base Transformer model use, "
            "and what are the key and value dimensions?"
        ),
        expected_answer="h=8 parallel attention heads, dk=dv=dmodel/h=64",
        verification_points=["8", "64"],
    ),
    BenchmarkQuestion(
        id="T3",
        experiment="trivial_rag",
        query=(
            "What optimizer and learning rate schedule was used to train "
            "the Transformer?"
        ),
        expected_answer=(
            "Adam optimizer with beta1=0.9, beta2=0.98, epsilon=1e-9. "
            "Warmup then inverse square root decay with warmup_steps=4000."
        ),
        verification_points=["Adam", "warmup", "4000"],
    ),
]

SUMMARY_QUESTIONS = [
    BenchmarkQuestion(
        id="S1",
        experiment="summary",
        query=(
            "Provide a comprehensive summary of this paper covering: "
            "(1) the key innovation, (2) the model architecture, "
            "(3) training setup, and (4) main experimental results."
        ),
        expected_answer=(
            "Transformer architecture based solely on attention mechanisms; "
            "encoder-decoder with multi-head self-attention; trained on WMT "
            "tasks; achieves 28.4 BLEU EN-DE and 41.8 BLEU EN-FR."
        ),
        verification_points=[
            "Transformer", "attention", "encoder", "decoder",
            "BLEU", "translation",
        ],
    ),
]

AGENTIC_QUESTIONS = [
    BenchmarkQuestion(
        id="A1",
        experiment="agentic",
        query=(
            "Explain all three types of attention used in the Transformer "
            "model. Then compare the computational complexity of "
            "self-attention versus recurrent and convolutional layers. "
            "Finally, describe the Transformer's results on English "
            "constituency parsing."
        ),
        expected_answer=(
            "Three attention types: encoder-decoder, encoder self-attention, "
            "masked decoder self-attention. Complexity: self-attention "
            "O(n^2*d), recurrent O(n*d^2). Parsing: 91.3 F1 WSJ-only, "
            "92.7 semi-supervised."
        ),
        verification_points=[
            "encoder-decoder", "self-attention", "masked",
            "constituency parsing",
        ],
    ),
    BenchmarkQuestion(
        id="A2",
        experiment="agentic",
        query=(
            "What types of regularization were used during training? "
            "How do the model variation experiments (Table 3) show "
            "the effect of the number of attention heads, model size, "
            "and dropout on translation quality?"
        ),
        expected_answer=(
            "Residual dropout (Pdrop=0.1) and label smoothing (eps=0.1). "
            "Single-head 0.9 BLEU worse; bigger models better; dropout helpful."
        ),
        verification_points=[
            "dropout", "label smoothing", "attention heads",
        ],
    ),
]


# ── PDF Text Extraction ─────────────────────────────────────────────────


def extract_pdf_text(path: str) -> str:
    doc = fitz.open(path)
    return "\n\n".join(page.get_text("text").strip() for page in doc)


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


# ── LLM Helpers ─────────────────────────────────────────────────────────

ANSWER_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY "
    "the provided context. Be specific and include numbers, names, and "
    "details when available. If the context is insufficient, say so."
)

SECTION_PICKER_PROMPT = (
    "Given a question and a table of contents, identify which sections "
    "are most relevant. Return ONLY a JSON array of section title strings."
)


async def llm_call(
    system: str, user: str, config: BenchmarkConfig, usage: TokenUsage
) -> str:
    _cb.set_target(usage)
    try:
        kwargs: Dict[str, Any] = {
            "model": config.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.api_base:
            kwargs["api_base"] = config.api_base
        resp = await litellm.acompletion(**kwargs)
        await asyncio.sleep(0.05)  # let callback fire
        return resp.choices[0].message.content or ""
    finally:
        _cb.set_target(None)


# ── Cleanup helpers ─────────────────────────────────────────────────────


def _cleanup_paths(*paths: str):
    for p in paths:
        path = Path(p)
        if path.exists():
            shutil.rmtree(path)


def _resolve_mentat_embedding(model: str) -> str:
    """Ensure the embedding model name is in mentat's expected format."""
    # Already in openai/openai/... format
    if model.count("/") >= 2:
        return model
    parts = model.split("/")
    if len(parts) == 2 and parts[0] == "openai":
        return f"openai/openai/{parts[1]}"
    return model


def _openai_model_name(model: str) -> str:
    """Strip 'openai/' prefix for the raw OpenAI client."""
    return model.replace("openai/", "", 1)


# ── System: Naive ───────────────────────────────────────────────────────


async def run_naive(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
    experiment: str,
) -> ExperimentResult:
    """Read the whole file, send everything as LLM context."""
    result = ExperimentResult(experiment=experiment, system="naive")

    t0 = time.perf_counter()
    full_text = extract_pdf_text(config.paper_path)
    result.indexing.wall_time = time.perf_counter() - t0

    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)
        context = f"Full paper text:\n\n{full_text}"
        qr.context_tokens_est = estimate_tokens(context)

        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    return result


# ── System: LanceDB RAG ────────────────────────────────────────────────


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    chunks, start, stride = [], 0, max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += stride
    return chunks


async def _embed_via_litellm(
    texts: List[str], config: BenchmarkConfig, usage: TokenUsage,
) -> List[List[float]]:
    """Embed texts using litellm (consistent with mentat)."""
    _cb.set_target(usage)
    kwargs: Dict[str, Any] = {"model": config.embedding_model, "input": texts}
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    resp = await litellm.aembedding(**kwargs)
    await asyncio.sleep(0.05)  # let callback fire
    return [item["embedding"] for item in resp.data]


async def run_lancedb(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
    experiment: str,
) -> ExperimentResult:
    """Standard RAG: chunk, embed, vector search."""
    import lancedb
    import pyarrow as pa

    result = ExperimentResult(experiment=experiment, system="lancedb")
    _cleanup_paths(config.lancedb_db_path)

    # ── Index ──
    t0 = time.perf_counter()
    full_text = extract_pdf_text(config.paper_path)
    chunks = _chunk_text(
        full_text, config.lancedb_chunk_size, config.lancedb_chunk_overlap
    )

    # Embed all chunks via litellm (batched)
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), 100):
        batch = chunks[i : i + 100]
        vecs = await _embed_via_litellm(batch, config, result.indexing)
        vectors.extend(vecs)

    db = lancedb.connect(config.lancedb_db_path)
    tbl = db.create_table(
        "chunks",
        data=pa.table(
            {
                "chunk_id": list(range(len(chunks))),
                "text": chunks,
                "vector": vectors,
            }
        ),
    )
    tbl.create_fts_index("text")
    result.indexing.wall_time = time.perf_counter() - t0

    # ── Query ──
    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)

        t0 = time.perf_counter()
        q_vecs = await _embed_via_litellm([q.query], config, qr.retrieval)
        q_vec = q_vecs[0]

        rows = (
            tbl.search(query_type="hybrid")
            .vector(q_vec)
            .text(q.query)
            .limit(config.top_k)
            .to_pandas()
        )
        qr.retrieval.wall_time = time.perf_counter() - t0

        context = "\n\n---\n\n".join(
            f"[Chunk {row['chunk_id']}]\n{row['text']}"
            for _, row in rows.iterrows()
        )
        qr.context_tokens_est = estimate_tokens(context)

        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    _cleanup_paths(config.lancedb_db_path)
    return result


# ── System: Mentat ──────────────────────────────────────────────────────


def _make_mentat(config: BenchmarkConfig):
    """Create fresh Mentat instance."""
    from mentat.core.hub import Mentat
    from mentat.core.models import MentatConfig
    from mentat.core.telemetry import Telemetry

    _cleanup_paths(config.mentat_db_path, config.mentat_storage_dir)
    Mentat.reset()
    Telemetry._stats.clear()

    cfg = MentatConfig(
        db_path=config.mentat_db_path,
        storage_dir=config.mentat_storage_dir,
        embedding_model=_resolve_mentat_embedding(config.embedding_model),
        embedding_api_key=config.api_key,
        embedding_api_base=config.api_base,
        summary_model=config.chat_model,
        summary_api_key=config.api_key,
        summary_api_base=config.api_base,
    )
    return Mentat.get_instance(cfg), Telemetry


async def _teardown_mentat(config: BenchmarkConfig):
    from mentat.core.hub import Mentat
    from mentat.core.telemetry import Telemetry

    try:
        await Mentat.get_instance().shutdown()
    except Exception:
        pass
    Mentat.reset()
    Telemetry._stats.clear()
    _cleanup_paths(config.mentat_db_path, config.mentat_storage_dir)


def _extract_toc_text(info: Optional[Dict]) -> str:
    """Build readable ToC string from inspect data."""
    if not info:
        return ""
    probe = info.get("probe", {})
    structure = probe.get("structure", {})
    toc = structure.get("toc", [])
    if not toc:
        return ""
    lines = ["Table of Contents:"]
    for e in toc:
        indent = "  " * (e.get("level", 1) - 1)
        title = e.get("title", "")
        page = e.get("page", "")
        preview = e.get("preview", "")
        line = f"  {indent}{title}"
        if page:
            line += f" (p.{page})"
        if preview:
            line += f" -- {preview}"
        lines.append(line)
    return "\n".join(lines)


def _extract_topic_text(info: Optional[Dict]) -> str:
    """Extract title and first paragraph from inspect data."""
    if not info:
        return ""
    probe = info.get("probe", {})
    topic = probe.get("topic", {})
    parts = []
    if topic.get("title"):
        parts.append(f"Title: {topic['title']}")
    if topic.get("abstract"):
        parts.append(f"Abstract: {topic['abstract']}")
    if topic.get("first_paragraph"):
        parts.append(f"Opening: {topic['first_paragraph']}")
    return "\n".join(parts)


# ── Mentat: Trivial RAG ────────────────────────────────────────────────


async def run_mentat_trivial_rag(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
) -> ExperimentResult:
    """Mentat for factual questions: index + vector search."""
    result = ExperimentResult(experiment="trivial_rag", system="mentat")
    m, Tel = _make_mentat(config)
    await m.start()

    # Index (no summarization for trivial RAG — just embeddings)
    _cb.set_target(result.indexing)
    t0 = time.perf_counter()
    doc_ids = await m.add_batch(
        [config.paper_path], force=True, summarize=False
    )
    result.indexing.wall_time = time.perf_counter() - t0
    await _cb.set_target_async(None)

    # Query
    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)

        _cb.set_target(qr.retrieval)
        t0 = time.perf_counter()
        results = await m.search(q.query, top_k=config.top_k)
        qr.retrieval.wall_time = time.perf_counter() - t0
        await _cb.set_target_async(None)

        context = "\n\n---\n\n".join(
            f"[{r.section or 'N/A'}]\n{r.content}" for r in results
        )
        qr.context_tokens_est = estimate_tokens(context)

        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    await _teardown_mentat(config)
    return result


# ── Mentat: Summary (ToC-guided) ────────────────────────────────────────


async def run_mentat_summary_toc(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
) -> ExperimentResult:
    """Use mentat's ToC to guide summary generation (no chunk summaries)."""
    result = ExperimentResult(experiment="summary", system="mentat-toc")
    m, Tel = _make_mentat(config)
    await m.start()

    # Index without summarization (only probe + embed)
    _cb.set_target(result.indexing)
    t0 = time.perf_counter()
    doc_ids = await m.add_batch(
        [config.paper_path], force=True, summarize=False
    )
    result.indexing.wall_time = time.perf_counter() - t0
    await _cb.set_target_async(None)

    info = await m.inspect(doc_ids[0])
    toc_text = _extract_toc_text(info)
    topic_text = _extract_topic_text(info)

    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)

        # Context = topic info + ToC (very compact)
        context = f"{topic_text}\n\n{toc_text}"
        qr.context_tokens_est = estimate_tokens(context)

        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    await _teardown_mentat(config)
    return result


# ── Mentat: Summary (pre-gen summaries) ─────────────────────────────────


async def run_mentat_summary_full(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
) -> ExperimentResult:
    """Use mentat's pre-gen chunk summaries for rich summary."""
    result = ExperimentResult(experiment="summary", system="mentat-summaries")
    m, Tel = _make_mentat(config)
    await m.start()

    # Index WITH summarization
    _cb.set_target(result.indexing)
    t0 = time.perf_counter()
    doc_ids = await m.add_batch(
        [config.paper_path], force=True, summarize=True
    )
    result.indexing.wall_time = time.perf_counter() - t0
    await _cb.set_target_async(None)

    info = await m.inspect(doc_ids[0])
    toc_text = _extract_toc_text(info)
    topic_text = _extract_topic_text(info)

    # Collect chunk summaries
    summaries_text = ""
    if info and info.get("chunk_summaries"):
        parts = []
        for s in info["chunk_summaries"]:
            summary = s.get("summary", "")
            section = s.get("section", "N/A")
            if summary:
                parts.append(f"[{section}] {summary}")
        summaries_text = "\n".join(parts)

    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)

        context = f"{topic_text}\n\n{toc_text}\n\nSection Summaries:\n{summaries_text}"
        qr.context_tokens_est = estimate_tokens(context)

        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    await _teardown_mentat(config)
    return result


# ── Mentat: Agentic (two-step ToC-guided retrieval) ─────────────────────


async def run_mentat_agentic(
    config: BenchmarkConfig,
    questions: List[BenchmarkQuestion],
) -> ExperimentResult:
    """Two-step Q&A: (1) ToC → LLM picks sections, (2) use pre-gen summaries.

    This avoids multiple embedding searches — instead it uses the chunk
    summaries from inspect() filtered by the sections the LLM selected.
    Only falls back to vector search for the original query to fill gaps.
    """
    result = ExperimentResult(experiment="agentic", system="mentat")
    m, Tel = _make_mentat(config)
    await m.start()

    # Index with summaries (one-time cost, amortized across queries)
    _cb.set_target(result.indexing)
    t0 = time.perf_counter()
    doc_ids = await m.add_batch(
        [config.paper_path], force=True, summarize=True
    )
    result.indexing.wall_time = time.perf_counter() - t0
    await _cb.set_target_async(None)

    info = await m.inspect(doc_ids[0])
    toc_text = _extract_toc_text(info)
    topic_text = _extract_topic_text(info)

    # Pre-build section → summaries map from inspect data
    section_summaries: Dict[str, List[str]] = {}
    if info and info.get("chunk_summaries"):
        for s in info["chunk_summaries"]:
            sec = (s.get("section") or "").strip()
            summary = (s.get("summary") or "").strip()
            if sec and summary:
                section_summaries.setdefault(sec, []).append(summary)

    for q in questions:
        qr = QuestionResult(question_id=q.id, query=q.query)

        # Step 1: Present ToC, ask LLM which sections are relevant
        step1_usage = TokenUsage()
        t0 = time.perf_counter()
        sections_raw = await llm_call(
            SECTION_PICKER_PROMPT,
            f"Question: {q.query}\n\n{toc_text}",
            config,
            step1_usage,
        )
        qr.retrieval.wall_time += time.perf_counter() - t0
        qr.retrieval.llm_prompt_tokens += step1_usage.llm_prompt_tokens
        qr.retrieval.llm_completion_tokens += step1_usage.llm_completion_tokens

        # Parse section titles
        try:
            cleaned = sections_raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            picked = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            picked = []

        # Step 2: Gather pre-gen summaries for the picked sections
        # (no embedding calls needed — uses data already computed at index time)
        context_parts = [topic_text]
        picked_lower = {s.lower() for s in picked}
        for sec_name, summaries in section_summaries.items():
            if sec_name.lower() in picked_lower:
                merged = " ".join(summaries)
                context_parts.append(f"[{sec_name}]\n{merged}")

        # Also do ONE vector search with the original query to fill gaps
        _cb.set_target(qr.retrieval)
        t0 = time.perf_counter()
        search_results = await m.search(q.query, top_k=config.top_k)
        qr.retrieval.wall_time += time.perf_counter() - t0
        await _cb.set_target_async(None)

        for r in search_results:
            chunk_text = f"[{r.section or 'N/A'}]\n{r.content}"
            if r.summary:
                chunk_text += f"\nSummary: {r.summary}"
            context_parts.append(chunk_text)

        context = "\n\n---\n\n".join(context_parts)
        qr.context_tokens_est = estimate_tokens(context)

        # Step 3: Generate answer
        t0 = time.perf_counter()
        qr.answer = await llm_call(
            ANSWER_PROMPT,
            f"Context:\n{context}\n\nQuestion: {q.query}",
            config,
            qr.answer_gen,
        )
        qr.answer_gen.wall_time = time.perf_counter() - t0
        result.questions.append(qr)

    await _teardown_mentat(config)
    return result


# ── Report ──────────────────────────────────────────────────────────────


def _fmt(n: int) -> str:
    return f"{n:,}"


def _fmtt(s: float) -> str:
    return f"{s:.2f}s"


def print_report(
    all_results: Dict[str, List[ExperimentResult]], verbose: bool = False
):
    for exp_name, results in all_results.items():
        if not results:
            continue

        names = [r.system for r in results]
        col_w = max(22, max(len(n) for n in names) + 4)
        hdr_w = 26

        def row(label: str, values: List[str]):
            print(f"  {label:<{hdr_w}}", end="")
            for v in values:
                print(f"{v:>{col_w}}", end="")
            print()

        w = hdr_w + col_w * len(names) + 2
        print(f"\n{'=' * w}")
        print(f"  EXPERIMENT: {exp_name.upper()}")
        print(f"{'=' * w}")
        row("", names)
        print(f"{'-' * w}")

        # Indexing
        print("\n  INDEXING (one-time cost)")
        row("Time", [_fmtt(r.indexing.wall_time) for r in results])
        row("Embed Tokens", [_fmt(r.indexing.embedding_tokens) for r in results])
        row(
            "LLM Tokens",
            [_fmt(r.indexing.total_llm) for r in results],
        )

        # Queries
        nq = max(len(r.questions) for r in results) if results else 0
        if nq:
            print(f"\n  QUERIES ({nq} question(s))")
            row(
                "Embed Tokens",
                [
                    _fmt(sum(q.retrieval.embedding_tokens for q in r.questions))
                    for r in results
                ],
            )
            row(
                "LLM Prompt Tokens",
                [
                    _fmt(
                        sum(
                            q.retrieval.llm_prompt_tokens
                            + q.answer_gen.llm_prompt_tokens
                            for q in r.questions
                        )
                    )
                    for r in results
                ],
            )
            row(
                "LLM Completion Tokens",
                [
                    _fmt(
                        sum(
                            q.retrieval.llm_completion_tokens
                            + q.answer_gen.llm_completion_tokens
                            for q in r.questions
                        )
                    )
                    for r in results
                ],
            )
            row(
                "Context Tokens (est.)",
                [
                    _fmt(sum(q.context_tokens_est for q in r.questions))
                    for r in results
                ],
            )
            row(
                "Query Time",
                [
                    _fmtt(
                        sum(
                            q.retrieval.wall_time + q.answer_gen.wall_time
                            for q in r.questions
                        )
                    )
                    for r in results
                ],
            )

        # Totals
        print(f"\n  TOTALS")
        row(
            "Embed Tokens (all)",
            [_fmt(r.total_embedding_tokens) for r in results],
        )
        row("LLM Tokens (all)", [_fmt(r.total_llm_tokens) for r in results])
        row("Total Time", [_fmtt(r.total_time) for r in results])
        print(f"{'=' * w}")

        if verbose:
            for r in results:
                print(f"\n  [{r.system}] Answers:")
                for q in r.questions:
                    print(f"    {q.question_id}: {q.answer[:500]}")
                    print()


def save_results(
    all_results: Dict[str, List[ExperimentResult]], config: BenchmarkConfig
):
    payload: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "paper": config.paper_path,
            "embedding_model": config.embedding_model,
            "chat_model": config.chat_model,
            "top_k": config.top_k,
        },
        "experiments": {},
    }

    for exp_name, results in all_results.items():
        exp_data: Dict[str, Any] = {}
        for r in results:
            exp_data[r.system] = {
                "indexing": {
                    "wall_time": r.indexing.wall_time,
                    "embedding_tokens": r.indexing.embedding_tokens,
                    "llm_prompt_tokens": r.indexing.llm_prompt_tokens,
                    "llm_completion_tokens": r.indexing.llm_completion_tokens,
                },
                "totals": {
                    "embedding_tokens": r.total_embedding_tokens,
                    "llm_tokens": r.total_llm_tokens,
                    "time": r.total_time,
                },
                "questions": [
                    {
                        "id": q.question_id,
                        "query": q.query,
                        "answer": q.answer,
                        "context_tokens_est": q.context_tokens_est,
                        "retrieval": {
                            "embedding_tokens": q.retrieval.embedding_tokens,
                            "llm_tokens": q.retrieval.total_llm,
                            "wall_time": q.retrieval.wall_time,
                        },
                        "answer_gen": {
                            "llm_prompt_tokens": q.answer_gen.llm_prompt_tokens,
                            "llm_completion_tokens": q.answer_gen.llm_completion_tokens,
                            "wall_time": q.answer_gen.wall_time,
                        },
                    }
                    for q in r.questions
                ],
            }
        payload["experiments"][exp_name] = exp_data

    with open(config.output_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Main Runner ─────────────────────────────────────────────────────────


async def run_benchmark(experiments: List[str], config: BenchmarkConfig):
    all_results: Dict[str, List[ExperimentResult]] = {}

    if "trivial_rag" in experiments:
        print(f"\n{'━' * 60}")
        print(f"  Experiment 1: Trivial RAG")
        print(f"{'━' * 60}")
        results: List[ExperimentResult] = []

        print("  Running: mentat ...")
        results.append(await run_mentat_trivial_rag(config, TRIVIAL_RAG_QUESTIONS))

        print("  Running: lancedb ...")
        results.append(
            await run_lancedb(config, TRIVIAL_RAG_QUESTIONS, "trivial_rag")
        )

        print("  Running: naive ...")
        results.append(
            await run_naive(config, TRIVIAL_RAG_QUESTIONS, "trivial_rag")
        )

        all_results["trivial_rag"] = results

    if "summary" in experiments:
        print(f"\n{'━' * 60}")
        print(f"  Experiment 2: Summary")
        print(f"{'━' * 60}")
        results = []

        print("  Running: mentat-toc ...")
        results.append(await run_mentat_summary_toc(config, SUMMARY_QUESTIONS))

        print("  Running: mentat-summaries ...")
        results.append(await run_mentat_summary_full(config, SUMMARY_QUESTIONS))

        print("  Running: lancedb ...")
        results.append(
            await run_lancedb(config, SUMMARY_QUESTIONS, "summary")
        )

        print("  Running: naive ...")
        results.append(await run_naive(config, SUMMARY_QUESTIONS, "summary"))

        all_results["summary"] = results

    if "agentic" in experiments:
        print(f"\n{'━' * 60}")
        print(f"  Experiment 3: Agentic Scene")
        print(f"{'━' * 60}")
        results = []

        print("  Running: mentat ...")
        results.append(await run_mentat_agentic(config, AGENTIC_QUESTIONS))

        print("  Running: lancedb ...")
        results.append(
            await run_lancedb(config, AGENTIC_QUESTIONS, "agentic")
        )

        print("  Running: naive ...")
        results.append(await run_naive(config, AGENTIC_QUESTIONS, "agentic"))

        all_results["agentic"] = results

    print_report(all_results, verbose=config.verbose)
    save_results(all_results, config)
    print(f"\nResults saved to {config.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Mentat Benchmark: Mentat vs LanceDB vs Naive"
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["trivial_rag", "summary", "agentic", "all"],
        default=["all"],
    )
    parser.add_argument("--output", default="benchmarks/results.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--chat-model", default="")
    args = parser.parse_args()

    experiments = (
        ["trivial_rag", "summary", "agentic"]
        if "all" in args.experiments
        else args.experiments
    )

    config = BenchmarkConfig(
        output_file=args.output,
        top_k=args.top_k,
        verbose=args.verbose,
    )
    if args.chat_model:
        config.chat_model = args.chat_model

    print("Benchmark Configuration:")
    print(f"  Paper:           {config.paper_path}")
    print(f"  Embedding Model: {config.embedding_model}")
    print(f"  Chat Model:      {config.chat_model}")
    print(f"  Experiments:     {', '.join(experiments)}")
    print(f"  Top-K:           {config.top_k}")

    asyncio.run(run_benchmark(experiments, config))


if __name__ == "__main__":
    main()
