#!/usr/bin/env python3
"""Compare reranker latency across none/local/external modes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from mentat.core.reranker import CrossEncoderReranker, ExternalReranker


def load_pairs(path: Path) -> List[Tuple[str, List[str]]]:
    raw = json.loads(path.read_text())
    return [(item["query"], item["documents"]) for item in raw]


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="benchmarks/reranker_cases.json")
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["none", "local"],
        choices=["none", "local", "external"],
    )
    parser.add_argument("--external-url", default=os.getenv("MENTAT_RERANKER_API_BASE", ""))
    parser.add_argument("--external-key", default=os.getenv("MENTAT_RERANKER_API_KEY", ""))
    args = parser.parse_args()

    pairs = load_pairs(Path(args.cases))

    local = None
    if "local" in args.modes:
        local = CrossEncoderReranker(model=args.model)

    external = None
    if "external" in args.modes:
        external = ExternalReranker(
            model=args.model,
            api_key=args.external_key,
            api_base=args.external_url,
        )

    measurements: Dict[str, List[float]] = {mode: [] for mode in args.modes}

    for _ in range(args.iterations):
        for query, documents in pairs:
            pair_list = [(query, doc) for doc in documents]

            if "none" in args.modes:
                started = time.perf_counter()
                _ = documents
                measurements["none"].append((time.perf_counter() - started) * 1000.0)

            if "local" in args.modes and local is not None:
                started = time.perf_counter()
                import asyncio

                asyncio.run(local.score_pairs(pair_list))
                measurements["local"].append((time.perf_counter() - started) * 1000.0)

            if "external" in args.modes and external is not None:
                started = time.perf_counter()
                import asyncio

                asyncio.run(external.score_pairs(pair_list))
                measurements["external"].append((time.perf_counter() - started) * 1000.0)

    print(f"cases={len(pairs)} iterations={args.iterations} model={args.model}")
    for mode in args.modes:
        values = measurements[mode]
        print(
            f"{mode}: n={len(values)} "
            f"mean_ms={statistics.mean(values):.2f} "
            f"p50_ms={statistics.median(values):.2f} "
            f"p95_ms={percentile(values, 0.95):.2f}"
        )


if __name__ == "__main__":
    main()
