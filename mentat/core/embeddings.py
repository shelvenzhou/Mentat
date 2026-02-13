import abc
from typing import List, Optional

import litellm


# ── Embedding providers ─────────────────────────────────────────────────────


class BaseEmbedding(abc.ABC):
    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts. Default falls back to sequential single calls."""
        import asyncio
        return await asyncio.gather(*(self.embed(t) for t in texts))


class LiteLLMEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    async def embed(self, text: str) -> List[float]:
        kwargs = {"model": self.model, "input": [text]}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = await litellm.aembedding(**kwargs)
        return response.data[0]["embedding"]

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed all texts in a single API call."""
        if not texts:
            return []
        kwargs = {"model": self.model, "input": texts}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["api_base"] = self.api_base
        response = await litellm.aembedding(**kwargs)
        # Sort by index to preserve input order
        sorted_data = sorted(response.data, key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]


class EmbeddingRegistry:
    _providers = {"litellm": LiteLLMEmbedding}

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseEmbedding:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown embedding provider: {name}")
        return provider_cls(**kwargs)
