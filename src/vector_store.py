"""Vector index contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from semantic_types import CacheEntry, CacheKey, CacheMetadata, Embedding, Query, Score


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    cache_key: CacheKey
    embedding: Embedding
    score: Score
    query: Query | None = None
    metadata: CacheMetadata = field(default_factory=CacheMetadata)
    payload: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Semantic candidate index owned by the oracle."""

    @abstractmethod
    def upsert(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, cache_key: CacheKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError
