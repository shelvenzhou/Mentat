"""Tests for reranker providers."""

import io
import json

import pytest

from mentat.core.reranker import ExternalReranker


class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._buffer = io.BytesIO(json.dumps(payload).encode("utf-8"))

    def read(self) -> bytes:
        return self._buffer.read()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_external_reranker_restores_scores_by_result_index(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=30):
        captured["url"] = request.full_url
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTPResponse(
            {
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.2},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    reranker = ExternalReranker(
        model="BAAI/bge-reranker-v2-m3",
        api_key="secret",
        api_base="https://example.com/v1/rerank",
    )
    scores = reranker._score_documents("query", ["doc a", "doc b"])

    assert scores == [0.2, 0.8]
    assert captured["url"] == "https://example.com/v1/rerank"
    assert captured["auth"] == "Bearer secret"
    assert captured["body"]["model"] == "BAAI/bge-reranker-v2-m3"
    assert captured["body"]["query"] == "query"
    assert captured["body"]["documents"] == ["doc a", "doc b"]
