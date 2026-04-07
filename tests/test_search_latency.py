"""Latency guardrails for retrieval, reranking, and heat-biased search."""

import time

import pytest


class FakeReranker:
    def __init__(self, scorer):
        self._scorer = scorer

    async def score_pairs(self, pairs):
        return [float(self._scorer(query, text)) for query, text in pairs]


VECTOR_SEARCH_BUDGET_MS = 50.0
RERANK_SEARCH_BUDGET_MS = 150.0
HEAT_BIAS_BUDGET_MS = 75.0


async def _seed_search_docs(mentat_instance, query: str, count: int = 50) -> None:
    query_vec = await mentat_instance.embeddings.embed(query)
    mentat_instance.storage.add_stub("seed", "seed.md", "", "", "{}")
    rows = []
    for idx in range(count):
        rows.append(
            {
                "chunk_id": f"seed_{idx}",
                "doc_id": "seed",
                "filename": "seed.md",
                "content": f"candidate {idx}",
                "summary": f"candidate {idx}",
                "section": f"S{idx}",
                "chunk_index": idx,
                "vector": query_vec,
            }
        )
    mentat_instance.storage.add_chunks(rows)


@pytest.mark.asyncio
async def test_vector_search_latency_budget(mentat_instance):
    query = "latency baseline"
    await _seed_search_docs(mentat_instance, query)

    start = time.perf_counter()
    results = await mentat_instance.search(query, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(results) == 10
    assert elapsed_ms < VECTOR_SEARCH_BUDGET_MS


@pytest.mark.asyncio
async def test_rerank_search_latency_budget(mentat_instance):
    query = "latency baseline rerank"
    await _seed_search_docs(mentat_instance, query)
    mentat_instance.config.reranker_enabled = True
    mentat_instance.config.reranker_weight = 0.85
    mentat_instance.config.reranker_top_n = 50
    mentat_instance.config.reranker_candidate_multiplier = 5
    mentat_instance.reranker = FakeReranker(
        scorer=lambda query, text: 1.0 if text.endswith("0") else 0.0
    )

    start = time.perf_counter()
    results = await mentat_instance.search(query, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(results) == 10
    assert elapsed_ms < RERANK_SEARCH_BUDGET_MS


@pytest.mark.asyncio
async def test_heat_bias_latency_budget(mentat_instance):
    query = "latency baseline heat"
    await _seed_search_docs(mentat_instance, query)
    mentat_instance.config.search_heat_weight = 0.25
    for idx in range(10):
        await mentat_instance.section_heat.record("seed", f"S{idx}", weight=idx + 1.0)

    start = time.perf_counter()
    results = await mentat_instance.search(query, top_k=10)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert len(results) == 10
    assert elapsed_ms < HEAT_BIAS_BUDGET_MS
