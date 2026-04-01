import abc
import asyncio
import logging
import random
from typing import List, Optional

import openai

logger = logging.getLogger(__name__)

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


class OpenAIEmbedding(BaseEmbedding):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        self.model = model
        self._client = openai.AsyncOpenAI(
            api_key=api_key or None,
            base_url=api_base or None,
        )

    async def embed(self, text: str) -> List[float]:
        response = await self._client.embeddings.create(
            model=self.model, input=[text]
        )
        return response.data[0].embedding

    # Per-text token limit (OpenAI text-embedding-3-*: 8191 tokens max per input)
    # Conservative to account for estimation error in _estimate_tokens()
    MAX_TOKENS_PER_TEXT = 6000

    # Max texts per API call.  OpenAI allows 2048; we use a smaller default
    # to keep individual request payloads manageable and allow concurrency.
    MAX_TEXTS_PER_BATCH = 100

    # Retry settings
    MAX_RETRIES = 4
    RETRY_BASE_DELAY = 1.0  # seconds
    RETRY_MAX_DELAY = 30.0  # seconds

    # Max concurrent batch requests to avoid overwhelming the API
    MAX_CONCURRENT_BATCHES = 4

    # Minimum seconds between consecutive batch requests
    BATCH_INTERVAL = 0.2

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed texts in batches that respect the model's limits.

        Batching strategy:
          * Each individual text is truncated to MAX_TOKENS_PER_TEXT if needed.
          * Texts are grouped into batches of up to MAX_TEXTS_PER_BATCH.
          * Batches are sent with bounded concurrency (MAX_CONCURRENT_BATCHES).
          * Each batch request retries with exponential backoff on transient errors.

        Returns:
            List of embeddings in same order as input texts
        """
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

        sem = asyncio.Semaphore(self.MAX_CONCURRENT_BATCHES)
        # Lock to enforce minimum interval between request starts
        interval_lock = asyncio.Lock()
        last_request_time = 0.0

        async def _embed_one_batch(batch_texts: List[str], batch_idx: int) -> List[List[float]]:
            nonlocal last_request_time
            async with sem:
                # Enforce minimum interval between requests
                async with interval_lock:
                    import time
                    now = time.monotonic()
                    wait = self.BATCH_INTERVAL - (now - last_request_time)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    last_request_time = time.monotonic()
                return await self._embed_one_batch_with_retry(
                    batch_texts, batch_idx, len(batches)
                )

        batch_results = await asyncio.gather(
            *(_embed_one_batch(batch, i) for i, batch in enumerate(batches))
        )

        # Flatten results while preserving order
        all_embeddings = []
        for batch_embeddings in batch_results:
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _embed_one_batch_with_retry(
        self,
        batch_texts: List[str],
        batch_idx: int,
        total_batches: int,
    ) -> List[List[float]]:
        """Send one embedding batch with exponential backoff retry."""
        batch_tokens = sum(_estimate_tokens(t) for t in batch_texts)
        text_lengths = [len(t) for t in batch_texts]
        logger.debug(
            f"Batch {batch_idx + 1}/{total_batches}: "
            f"{len(batch_texts)} texts, est. {batch_tokens} tokens, "
            f"total_chars={sum(text_lengths)}, "
            f"max_chars={max(text_lengths)}, min_chars={min(text_lengths)}, "
            f"model={self.model}, base_url={self._client.base_url}"
        )

        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES + 1):
            try:
                if attempt > 0:
                    delay = min(
                        self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        self.RETRY_MAX_DELAY,
                    )
                    # Add jitter to avoid thundering herd
                    delay *= 0.5 + random.random()
                    logger.warning(
                        f"Batch {batch_idx + 1}/{total_batches} retry {attempt}/{self.MAX_RETRIES} "
                        f"after {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)

                response = await self._client.embeddings.create(
                    model=self.model, input=batch_texts
                )
                sorted_data = sorted(response.data, key=lambda x: x.index)
                return [item.embedding for item in sorted_data]

            except (
                openai.InternalServerError,
                openai.RateLimitError,
                openai.APITimeoutError,
                openai.APIConnectionError,
            ) as e:
                last_exc = e
                logger.warning(
                    f"Batch {batch_idx + 1}/{total_batches} attempt {attempt + 1}/{self.MAX_RETRIES + 1} "
                    f"failed: {type(e).__name__}: {e} "
                    f"[{len(batch_texts)} texts, est. {batch_tokens} tokens, "
                    f"max_chars={max(text_lengths)}]"
                )
                if attempt == self.MAX_RETRIES:
                    logger.error(
                        f"Batch {batch_idx + 1}/{total_batches} gave up after "
                        f"{self.MAX_RETRIES + 1} attempts: {type(e).__name__}: {e}"
                    )
                    raise

        raise last_exc  # unreachable, but satisfies type checker


class EmbeddingRegistry:
    _providers = {"openai": OpenAIEmbedding}

    @classmethod
    def get_provider(cls, name: str, **kwargs) -> BaseEmbedding:
        provider_cls = cls._providers.get(name)
        if not provider_cls:
            raise ValueError(f"Unknown embedding provider: {name}")
        return provider_cls(**kwargs)
