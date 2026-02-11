import abc
from typing import List
import litellm


class BaseEmbedding(abc.ABC):
    @abc.abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass


class LiteLLMEmbedding(BaseEmbedding):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model

    async def embed(self, text: str) -> List[float]:
        response = await litellm.aembedding(model=self.model, input=[text])
        return response.data[0]["embedding"]


class EmbeddingRegistry:
    _providers = {"litellm": LiteLLMEmbedding}

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseEmbedding:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown embedding provider: {name}")
        return provider_cls(**kwargs)
