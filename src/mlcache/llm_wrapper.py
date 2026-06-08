"""High-level cache-first LLM wrapper.

`CachedLLM` orchestrates "check the cache, fall back to the LLM, write the
result back" around an `MLCache`. It deliberately does not implement any
semantic matching itself: every lookup, vector search, scoring, thresholding,
and hit/miss decision is delegated to `cache.lookup_with_decision`. The
wrapper only does cache-first orchestration, LLM fallback, cache writes on
miss, and structured response metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from mlcache.builder import MLCache
from mlcache.embeddings import EmbeddingProvider, HashingEmbeddingProvider
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, CacheMetadata, Query, Response


@dataclass(frozen=True)
class LLMResponse:
    """Raw output from an `LLMClient`."""

    text: str
    raw: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Minimal interface a language model backend must satisfy."""

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        ...


@dataclass
class MockLLM:
    """Deterministic, offline stand-in for a real LLM backend.

    Makes no network calls and requires no API key; it simply renders
    `response_template` with the prompt. Useful for exercising the cache
    wiring before integrating a real provider.
    """

    response_template: str = "mock response for: {prompt}"

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        return LLMResponse(
            text=self.response_template.format(prompt=prompt),
            metadata={"provider": "mock", "kwargs": kwargs},
        )


@dataclass(frozen=True)
class CachedLLMResponse:
    """Structured result of `CachedLLM.generate`."""

    text: str
    source: Literal["cache", "llm"]
    cache_key: str | None
    score: float | None
    threshold: float | None
    metadata: dict[str, Any] = field(default_factory=dict)


class CachedLLM:
    """Cache-first orchestration around an `LLMClient`.

    All semantic-matching decisions (lookup, vector search, scoring,
    thresholding, hit/miss) belong to the `MLCache`/oracle; this wrapper only
    decides whether to consult the cache, when to call the LLM, and whether to
    persist the LLM's answer.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        cache: MLCache,
        namespace: str | None = None,
        cache_reads: bool = True,
        cache_writes: bool = True,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.namespace = namespace
        self.cache_reads = bool(cache_reads)
        self.cache_writes = bool(cache_writes)
        self._embeddings = embeddings if embeddings is not None else HashingEmbeddingProvider()

    def generate(self, prompt: str, **kwargs: Any) -> CachedLLMResponse:
        embedding = self._embeddings.embed(prompt)

        if self.cache_reads:
            result = self.cache.lookup_with_decision(CacheLookup(query=Query(prompt), embedding=embedding))
            decision = result.decision
            if decision.accepted and result.response is not None:
                return CachedLLMResponse(
                    text=str(result.response),
                    source="cache",
                    cache_key=str(decision.cache_key) if decision.cache_key is not None else None,
                    score=float(decision.score) if decision.score is not None else None,
                    threshold=float(decision.threshold) if decision.threshold is not None else None,
                    metadata=dict(result.metadata),
                )

        llm_response = self.llm.generate(prompt, **kwargs)
        cache_key = self._cache_key_for(prompt)

        if self.cache_writes:
            self.cache.put(
                CacheEntry(
                    cache_key=cache_key,
                    query=Query(prompt),
                    response=Response(llm_response.text),
                    embedding=embedding,
                    metadata=CacheMetadata(namespace=self.namespace),
                )
            )

        return CachedLLMResponse(
            text=llm_response.text,
            source="llm",
            cache_key=str(cache_key) if self.cache_writes else None,
            score=None,
            threshold=None,
            metadata={"llm": "mock", "generation_kwargs": dict(kwargs)},
        )

    def _cache_key_for(self, prompt: str) -> CacheKey:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
        prefix = f"{self.namespace}:" if self.namespace else ""
        return CacheKey(f"{prefix}{digest}")


__all__ = [
    "CachedLLM",
    "CachedLLMResponse",
    "LLMClient",
    "LLMResponse",
    "MockLLM",
]
