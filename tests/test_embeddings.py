from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mentat.core.embeddings import EmbeddingRegistry, LiteLLMEmbedding


def _mock_embedding_response(n: int):
    return SimpleNamespace(
        data=[{"index": i, "embedding": [float(i)]} for i in range(n)]
    )


def test_registry_get_known_provider():
    provider = EmbeddingRegistry.get_provider("litellm", model="openai/text-embedding-3-small")
    assert isinstance(provider, LiteLLMEmbedding)


def test_registry_get_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        EmbeddingRegistry.get_provider("unknown")


@pytest.mark.asyncio
async def test_embed_batch_empty():
    emb = LiteLLMEmbedding(model="openai/text-embedding-3-small")
    out = await emb.embed_batch([])
    assert out == []


@pytest.mark.asyncio
async def test_embed_batch_truncation(monkeypatch):
    emb = LiteLLMEmbedding(model="openai/text-embedding-3-small")
    emb.MAX_TOKENS_PER_TEXT = 2  # ~6 chars
    emb.MAX_TEXTS_PER_BATCH = 10

    mock = AsyncMock(return_value=_mock_embedding_response(1))
    monkeypatch.setattr("mentat.core.embeddings.litellm.aembedding", mock)

    text = "1234567890"
    await emb.embed_batch([text])

    sent = mock.call_args.kwargs["input"][0]
    assert len(sent) == 6


@pytest.mark.asyncio
async def test_embed_batch_preserves_order(monkeypatch):
    emb = LiteLLMEmbedding(model="openai/text-embedding-3-small")

    response = SimpleNamespace(
        data=[
            {"index": 2, "embedding": [2.0]},
            {"index": 0, "embedding": [0.0]},
            {"index": 1, "embedding": [1.0]},
        ]
    )
    monkeypatch.setattr(
        "mentat.core.embeddings.litellm.aembedding",
        AsyncMock(return_value=response),
    )

    out = await emb.embed_batch(["a", "b", "c"])
    assert out == [[0.0], [1.0], [2.0]]
