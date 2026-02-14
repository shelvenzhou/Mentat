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

    async def embed_batch(self, texts: List[str], max_tokens_per_batch: int = 6000) -> List[List[float]]:
        """Embed texts in batches that respect the model's context window.

        Args:
            texts: List of texts to embed
            max_tokens_per_batch: Maximum tokens per API call (default 6000 to leave headroom)

        Returns:
            List of embeddings in same order as input texts
        """
        if not texts:
            return []

        # Single text or small batch - send as-is
        if len(texts) == 1:
            kwargs = {"model": self.model, "input": texts}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.api_base:
                kwargs["api_base"] = self.api_base
            response = await litellm.aembedding(**kwargs)
            return [response.data[0]["embedding"]]

        # Batch texts to respect context window
        batches = []
        current_batch = []
        current_tokens = 0

        for i, text in enumerate(texts):
            text_tokens = _estimate_tokens(text)

            # If a single text is too large, warn and truncate it
            if text_tokens > max_tokens_per_batch:
                import logging
                logger = logging.getLogger(__name__)
                # Truncate to fit (rough approximation: chars = tokens * 4)
                max_chars = int(max_tokens_per_batch * 4)
                original_len = len(text)
                text = text[:max_chars]
                text_tokens = max_tokens_per_batch - 100  # Leave some headroom
                logger.warning(
                    f"Text {i} too large ({original_len} chars, est. {_estimate_tokens(texts[i])} tokens), "
                    f"truncated to {max_chars} chars"
                )

            # If adding this text would exceed limit, start new batch
            if current_batch and current_tokens + text_tokens > max_tokens_per_batch:
                batches.append(current_batch)
                current_batch = [text]
                current_tokens = text_tokens
            else:
                current_batch.append(text)
                current_tokens += text_tokens

        # Add final batch
        if current_batch:
            batches.append(current_batch)

        # Process batches concurrently
        import asyncio
        import logging
        logger = logging.getLogger(__name__)

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
