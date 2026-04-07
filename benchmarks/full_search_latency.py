#!/usr/bin/env python3
"""Compare full mentat.search() latency across reranker modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Sequence

from mentat.core.hub import Mentat
from mentat.core.models import MentatConfig
from mentat.core.reranker import ExternalReranker


class FakeEmbedding:
    def _vec(self, text: str):
        base = sum(ord(c) for c in text) % 97
        return [float((base + i) % 17) / 17.0 for i in range(8)]

    async def embed(self, text: str):
        return self._vec(text)

    async def embed_batch(self, texts):
        return [self._vec(t) for t in texts]


def load_cases(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


async def build_mentat(mode: str, model: str, external_url: str, external_key: str) -> Mentat:
    temp_dir = tempfile.TemporaryDirectory()
    config = MentatConfig(
        db_path=str(Path(temp_dir.name) / "db"),
        storage_dir=str(Path(temp_dir.name) / "files"),
        reranker_enabled=(mode != "none"),
        reranker_provider=("cross_encoder" if mode == "local" else "external"),
        reranker_model=model,
        reranker_api_base=external_url,
        reranker_api_key=external_key,
        reranker_top_n=20,
        reranker_candidate_multiplier=5,
    )
    mentat = Mentat(config)
    mentat._temp_dir = temp_dir  # type: ignore[attr-defined]
    mentat.embeddings = FakeEmbedding()
    if mode == "external":
        mentat.reranker = ExternalReranker(
            model=model,
            api_base=external_url,
            api_key=external_key,
        )
    return mentat


async def seed_cases(mentat: Mentat, cases: List[dict]) -> None:
    for idx, case in enumerate(cases):
        query_vec = await mentat.embeddings.embed(case["query"])
        mentat.storage.add_stub(f"doc_{idx}", f"doc_{idx}.md", "", "", "{}")
        chunks = []
        for doc_idx, content in enumerate(case["documents"]):
            vector = query_vec if doc_idx == case["expected_top_index"] else [v + 3.0 for v in query_vec]
            chunks.append(
                {
                    "chunk_id": f"doc_{idx}_{doc_idx}",
                    "doc_id": f"doc_{idx}",
                    "filename": f"doc_{idx}.md",
                    "content": content,
                    "summary": content,
                    "section": f"S{doc_idx}",
                    "chunk_index": doc_idx,
                    "vector": vector,
                }
            )
        mentat.storage.add_chunks(chunks)


async def measure_mode(
    mode: str,
    cases: List[dict],
    iterations: int,
    model: str,
    external_url: str,
    external_key: str,
) -> List[float]:
    mentat = await build_mentat(mode, model, external_url, external_key)
    await seed_cases(mentat, cases)

    timings: List[float] = []
    for _ in range(iterations):
        for case in cases:
            started = time.perf_counter()
            await mentat.search(case["query"], top_k=2)
            timings.append((time.perf_counter() - started) * 1000.0)

    await mentat.shutdown()
    Mentat.reset()
    temp_dir = getattr(mentat, "_temp_dir", None)
    if temp_dir is not None:
        temp_dir.cleanup()
    return timings


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="benchmarks/reranker_cases.json")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "local"],
        choices=["none", "local", "external"],
    )
    parser.add_argument("--external-url", default=os.getenv("MENTAT_RERANKER_API_BASE", ""))
    parser.add_argument("--external-key", default=os.getenv("MENTAT_RERANKER_API_KEY", ""))
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    print(f"cases={len(cases)} iterations={args.iterations} model={args.model}")

    for mode in args.modes:
        values = await measure_mode(
            mode,
            cases,
            args.iterations,
            args.model,
            args.external_url,
            args.external_key,
        )
        print(
            f"{mode}: n={len(values)} "
            f"mean_ms={statistics.mean(values):.2f} "
            f"p50_ms={statistics.median(values):.2f} "
            f"p95_ms={percentile(values, 0.95):.2f}"
        )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
