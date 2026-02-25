import abc
from typing import List, Optional

import litellm

# Token estimation for batching
def _estimate_tokens(text: str) -> int:
    """Estimate token count from text.

    Uses character-based estimation which works better for JSON/code.
    Rule of thumb: 1 token ≈ 3-4 characters for most content.
    Using 3 chars/token for conservative estimate.
    """
    return int(len(text) / 3)


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

    # Per-text token limit (OpenAI text-embedding-3-*: 8191 tokens max per input)
    # Conservative to account for estimation error in _estimate_tokens()
    MAX_TOKENS_PER_TEXT = 6000

    # Max texts per API call.  OpenAI allows 2048; we use a smaller default
    # to keep individual request payloads manageable and allow concurrency.
    MAX_TEXTS_PER_BATCH = 100

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in batches that respect the model's limits.

        Batching strategy:
          * Each individual text is truncated to MAX_TOKENS_PER_TEXT if needed.
          * Texts are grouped into batches of up to MAX_TEXTS_PER_BATCH.
          * All batches are sent concurrently.

        Returns:
            List of embeddings in same order as input texts
        """
        import asyncio
        import logging
        logger = logging.getLogger(__name__)

        if not texts:
            return []

        # Truncate oversized individual texts
        processed: List[str] = []
        for i, text in enumerate(texts):
            text_tokens = _estimate_tokens(text)
            if text_tokens > self.MAX_TOKENS_PER_TEXT:
                max_chars = int(self.MAX_TOKENS_PER_TEXT * 3)
                logger.warning(
                    f"Text {i} too large (est. {text_tokens} tokens), "
                    f"truncated to ~{self.MAX_TOKENS_PER_TEXT} tokens"
                )
                processed.append(text[:max_chars])
            else:
                processed.append(text)

        # Build batches by count
        batches = [
            processed[i : i + self.MAX_TEXTS_PER_BATCH]
            for i in range(0, len(processed), self.MAX_TEXTS_PER_BATCH)
        ]

        async def _embed_one_batch(batch_texts: List[str], batch_idx: int) -> List[List[float]]:
            batch_tokens = sum(_estimate_tokens(t) for t in batch_texts)
            logger.debug(
                f"Embedding batch {batch_idx + 1}/{len(batches)}: "
                f"{len(batch_texts)} texts, est. {batch_tokens} tokens"
            )
            kwargs = {"model": self.model, "input": batch_texts}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.aembedding(**kwargs)
            # Sort by index to preserve order within batch
            sorted_data = sorted(response.data, key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]

        batch_results = await asyncio.gather(
            *(_embed_one_batch(batch, i) for i, batch in enumerate(batches))
        )

        # Flatten results while preserving order
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


class EmbeddingRegistry:
    _providers = {"litellm": LiteLLMEmbedding}

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseEmbedding:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown embedding provider: {name}")
        return provider_cls(**kwargs)
