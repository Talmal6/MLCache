"""In-memory vector retrieval backend."""

from __future__ import annotations

from math import sqrt

from mlcache.retrieval.vector_store import VectorSearchResult, VectorStore
from mlcache.semantic_types import CacheEntry, CacheKey, Embedding, Score


class InMemoryVectorStore(VectorStore):
    """Deterministic in-memory vector store with cosine or dot similarity."""

    def __init__(self, *, similarity: str = "cosine") -> None:
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")
        self.similarity = similarity
        self._entries: dict[CacheKey, VectorSearchResult] = {}

    def upsert(self, entry: CacheEntry) -> None:
        self._entries[CacheKey(str(entry.cache_key))] = VectorSearchResult(
            cache_key=CacheKey(str(entry.cache_key)),
            embedding=tuple(float(value) for value in entry.embedding),
            score=Score(1.0),
            query=entry.query,
            metadata=entry.metadata,
            payload={"response": entry.response},
        )

    def delete(self, cache_key: CacheKey) -> None:
        self._entries.pop(CacheKey(str(cache_key)), None)

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        result = self._entries.get(CacheKey(str(cache_key)))
        if result is None:
            return None
        return self._copy_result(result)

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
    ) -> list[VectorSearchResult]:
        query_embedding = tuple(float(value) for value in embedding)
        scored = []
        for result in self._entries.values():
            if namespace is not None and result.metadata.namespace != namespace:
                continue
            score = self._score(query_embedding, result.embedding)
            scored.append((score, str(result.cache_key), result))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            self._copy_result(result, score=Score(float(score)))
            for score, _, result in scored[: max(0, int(top_k))]
        ]

    def clear(self) -> None:
        self._entries.clear()

    def size(self) -> int:
        return len(self._entries)

    def _score(self, left: tuple[float, ...], right: Embedding) -> float:
        right_values = tuple(float(value) for value in right)
        if self.similarity == "dot":
            return sum(l * r for l, r in zip(left, right_values))
        left_norm = sqrt(sum(value * value for value in left))
        right_norm = sqrt(sum(value * value for value in right_values))
        if left_norm <= 0.0 or right_norm <= 0.0:
            return 0.0
        return sum(l * r for l, r in zip(left, right_values)) / (left_norm * right_norm)

    @staticmethod
    def _copy_result(result: VectorSearchResult, *, score: Score | None = None) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=CacheKey(str(result.cache_key)),
            embedding=tuple(float(value) for value in result.embedding),
            score=score if score is not None else Score(float(result.score)),
            query=result.query,
            metadata=result.metadata,
            payload=dict(result.payload),
        )


__all__ = ["InMemoryVectorStore"]
