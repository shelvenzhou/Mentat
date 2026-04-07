"""Tests for cross-encoder reranking and heat-based search bias."""

import time

import pytest


class FakeReranker:
    def __init__(self, scorer):
        self._scorer = scorer

    async def score_pairs(self, pairs):
        return [float(self._scorer(query, text)) for query, text in pairs]


def _chunk(
    *,
    chunk_id: str,
    doc_id: str,
    content: str,
    section: str,
    vector: list[float],
) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "filename": f"{doc_id}.md",
        "content": content,
        "summary": content,
        "section": section,
        "chunk_index": 0,
        "vector": vector,
    }


@pytest.mark.asyncio
async def test_cross_encoder_rerank_can_flip_semantic_order(mentat_instance):
    mentat_instance.config.reranker_enabled = True
    mentat_instance.config.reranker_weight = 1.0
    mentat_instance.config.reranker_top_n = 10
    mentat_instance.config.reranker_candidate_multiplier = 5
    mentat_instance.reranker = FakeReranker(
        scorer=lambda query, text: 10.0 if "技术会议" in text else 1.0
    )

    query = "我在干什么"
    query_vec = await mentat_instance.embeddings.embed(query)
    far_vec = [v + 3.0 for v in query_vec]

    mentat_instance.storage.add_stub("d1", "d1.md", "", "", "{}")
    mentat_instance.storage.add_stub("d2", "d2.md", "", "", "{}")
    mentat_instance.storage.add_chunks(
        [
            _chunk(
                chunk_id="d1_0",
                doc_id="d1",
                content="我在xxx酒店",
                section="地点",
                vector=query_vec,
            ),
            _chunk(
                chunk_id="d2_0",
                doc_id="d2",
                content="我正在开技术会议",
                section="活动",
                vector=far_vec,
            ),
        ]
    )

    results = await mentat_instance.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].content == "我正在开技术会议"
    assert results[1].content == "我在xxx酒店"


@pytest.mark.asyncio
async def test_time_decay_heat_bias_is_independent_from_reranker(mentat_instance, monkeypatch):
    mentat_instance.config.search_heat_weight = 0.5
    mentat_instance.config.reranker_enabled = False
    mentat_instance.reranker = None
    mentat_instance.section_heat._half_life = 10.0

    query = "status"
    query_vec = await mentat_instance.embeddings.embed(query)

    mentat_instance.storage.add_stub("d1", "d1.md", "", "", "{}")
    mentat_instance.storage.add_stub("d2", "d2.md", "", "", "{}")
    mentat_instance.storage.add_chunks(
        [
            _chunk(
                chunk_id="d1_0",
                doc_id="d1",
                content="old hot section",
                section="old",
                vector=query_vec,
            ),
            _chunk(
                chunk_id="d2_0",
                doc_id="d2",
                content="recent hot section",
                section="recent",
                vector=query_vec,
            ),
        ]
    )

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    await mentat_instance.section_heat.record("d1", "old", weight=8.0)
    fake_time[0] += 40.0
    await mentat_instance.section_heat.record("d2", "recent", weight=4.0)

    results = await mentat_instance.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].section == "recent"
    assert results[1].section == "old"


@pytest.mark.asyncio
async def test_heat_bias_can_promote_result_outside_initial_top_k(mentat_instance):
    mentat_instance.config.search_heat_weight = 1.0
    mentat_instance.config.reranker_enabled = False
    mentat_instance.reranker = None

    query = "priority"
    query_vec = await mentat_instance.embeddings.embed(query)

    mentat_instance.storage.add_stub("seed", "seed.md", "", "", "{}")
    rows = []
    for idx in range(7):
        rows.append(
            _chunk(
                chunk_id=f"seed_{idx}",
                doc_id="seed",
                content=f"candidate {idx}",
                section=f"S{idx}",
                vector=[v + (idx * 0.01) for v in query_vec],
            )
        )
        rows[-1]["chunk_index"] = idx
    mentat_instance.storage.add_chunks(rows)

    await mentat_instance.section_heat.record("seed", "S5", weight=50.0)

    results = await mentat_instance.search(query, top_k=5)

    assert len(results) == 5
    assert results[0].section == "S5"


@pytest.mark.asyncio
async def test_cross_encoder_rerank_english_activity_query(mentat_instance):
    mentat_instance.config.reranker_enabled = True
    mentat_instance.config.reranker_weight = 1.0
    mentat_instance.reranker = FakeReranker(
        scorer=lambda query, text: 10.0
        if query == "What am I doing?" and "technical meeting" in text
        else 1.0
    )

    query = "What am I doing?"
    query_vec = await mentat_instance.embeddings.embed(query)
    far_vec = [v + 3.0 for v in query_vec]

    mentat_instance.storage.add_stub("e1", "e1.md", "", "", "{}")
    mentat_instance.storage.add_stub("e2", "e2.md", "", "", "{}")
    mentat_instance.storage.add_chunks(
        [
            _chunk(
                chunk_id="e1_0",
                doc_id="e1",
                content="I am at the hotel",
                section="location",
                vector=query_vec,
            ),
            _chunk(
                chunk_id="e2_0",
                doc_id="e2",
                content="I am having a technical meeting",
                section="activity",
                vector=far_vec,
            ),
        ]
    )

    results = await mentat_instance.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].content == "I am having a technical meeting"
