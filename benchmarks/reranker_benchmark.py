#!/usr/bin/env python3
"""Benchmark reranker behavior on short Chinese retrieval cases.

Usage:
    UV_CACHE_DIR=/tmp/uv-cache HF_HOME=/tmp/hf-home \
      uv run python benchmarks/reranker_benchmark.py

    UV_CACHE_DIR=/tmp/uv-cache HF_HOME=/tmp/hf-home \
      uv run python benchmarks/reranker_benchmark.py --strategies raw local
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List


CASES_PATH = Path("benchmarks/reranker_cases.json")
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class Case:
    id: str
    query: str
    documents: List[str]
    expected_top_index: int
    note: str = ""


def load_cases(path: Path) -> List[Case]:
    raw = json.loads(path.read_text())
    return [Case(**item) for item in raw]


def lexical_baseline_scores(query: str, documents: List[str]) -> List[float]:
    """Cheap baseline: count overlapping characters, with action/location hints."""
    scores: List[float] = []
    query_chars = set(query)
    wants_action = any(phrase in query for phrase in ("做什么", "干什么"))
    wants_location = any(phrase in query for phrase in ("什么地方", "在哪", "哪里"))
    for doc in documents:
        score = float(len(query_chars.intersection(set(doc))))
        if wants_action and any(tok in doc for tok in ("正在", "开", "谈", "做")):
            score += 2.0
        if wants_location and any(tok in doc for tok in ("在", "酒店", "机场", "家", "公司")):
            score += 2.0
        scores.append(score)
    return scores


def rank_from_scores(scores: List[float]) -> List[int]:
    return sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)


def evaluate_case(
    case: Case,
    strategy: str,
    score_fn: Callable[[str, List[str]], List[float]],
) -> Dict:
    final_query = case.query
    started = time.perf_counter()
    scores = score_fn(final_query, case.documents)
    latency_ms = (time.perf_counter() - started) * 1000.0
    ranking = rank_from_scores(scores)
    top_index = ranking[0]
    return {
        "case_id": case.id,
        "strategy": strategy,
        "query": case.query,
        "effective_query": final_query,
        "expected_top_index": case.expected_top_index,
        "actual_top_index": top_index,
        "correct": top_index == case.expected_top_index,
        "latency_ms": latency_ms,
        "scores": [float(s) for s in scores],
        "documents": case.documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the reranker only from local Hugging Face cache.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["raw", "local"],
        choices=["raw", "local"],
    )
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    results: List[Dict] = []

    local_model = None
    if any(s in args.strategies for s in ("local", "rewrite_local")):
        from sentence_transformers import CrossEncoder

        local_model = CrossEncoder(
            args.model,
            local_files_only=args.local_files_only,
        )

    def local_scores(query: str, documents: List[str]) -> List[float]:
        assert local_model is not None
        pairs = [(query, doc) for doc in documents]
        raw = local_model.predict(pairs, convert_to_numpy=False)
        return [float(x) for x in raw]

    strategy_map = {
        "raw": lexical_baseline_scores,
        "local": local_scores,
    }

    for strategy in args.strategies:
        score_fn = strategy_map[strategy]
        for case in cases:
            results.append(evaluate_case(case, strategy, score_fn))

    grouped: Dict[str, List[Dict]] = {}
    for row in results:
        grouped.setdefault(row["strategy"], []).append(row)

    print(f"Cases: {len(cases)}")
    print(f"Model: {args.model}")
    print()
    for strategy in args.strategies:
        rows = grouped[strategy]
        accuracy = sum(1 for r in rows if r["correct"]) / len(rows)
        p50 = statistics.median(r["latency_ms"] for r in rows)
        print(f"[{strategy}] accuracy={accuracy:.2%} p50_latency_ms={p50:.2f}")
        for row in rows:
            status = "PASS" if row["correct"] else "FAIL"
            print(
                f"  {status} {row['case_id']} top={row['actual_top_index']} "
                f"expected={row['expected_top_index']} "
                f"query={row['query']} effective={row['effective_query']}"
            )
        print()


if __name__ == "__main__":
    main()
