"""Cross-encoder reranking providers."""

import abc
import asyncio
import json
import urllib.request
from typing import List, Optional, Sequence, Tuple


class BaseReranker(abc.ABC):
    """Scores (query, document) pairs for reranking."""

    @abc.abstractmethod
    async def score_pairs(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        """Return one score per pair, higher = more relevant."""


class CrossEncoderReranker(BaseReranker):
    """Local cross-encoder reranker via sentence-transformers."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        del api_key, api_base
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ValueError(
                "CrossEncoder reranker requires the optional dependency "
                "'sentence-transformers'."
            ) from exc
        self._model = CrossEncoder(model)

    async def score_pairs(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        if not pairs:
            return []
        return await asyncio.to_thread(
            self._model.predict, list(pairs), convert_to_numpy=False
        )


class ExternalReranker(BaseReranker):
    """Remote HTTP reranker compatible with /v1/rerank style APIs."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        if not api_base:
            raise ValueError("External reranker requires api_base")
        self.model = model
        self.api_key = api_key or ""
        self.api_base = api_base

    async def score_pairs(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [text for _, text in pairs]
        return await asyncio.to_thread(
            self._score_documents, query, documents
        )

    def _score_documents(self, query: str, documents: List[str]) -> List[float]:
        payload = json.dumps(
            {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.api_base,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        scores = [0.0 for _ in documents]
        for item in data.get("results", []):
            index = int(item.get("index", -1))
            if 0 <= index < len(scores):
                scores[index] = float(item.get("relevance_score", 0.0))
        return scores


class RerankerRegistry:
    _providers = {
        "cross_encoder": CrossEncoderReranker,
        "external": ExternalReranker,
    }

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseReranker:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown reranker provider: {name}")
        return provider_cls(**kwargs)
