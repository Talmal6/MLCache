"""In-memory vector retrieval backend."""

from __future__ import annotations

import numpy as np

from mlcache.retrieval.vector_store import VectorSearchResult, VectorStore
from mlcache.semantic_types import CacheEntry, CacheKey, CacheMetadata, Embedding, Score


class InMemoryVectorStore(VectorStore):
    """Deterministic in-memory vector store with cosine or dot similarity.

    Embeddings are kept in a dense numpy matrix (grown by doubling) so
    `search` scores every candidate with a single vectorized matmul instead
    of a per-entry Python loop -- the latter made `search` (and therefore
    every `handle()` call) cost O(store_size * dim) in pure Python, which
    dominates runtime once the store and embedding dimension grow large.
    """

    def __init__(self, *, similarity: str = "cosine") -> None:
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")
        self.similarity = similarity
        self._entries: dict[str, VectorSearchResult] = {}
        self._key_to_idx: dict[str, int] = {}
        self._idx_to_key: list[str] = []
        self._namespaces: list[str | None] = []
        self._matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._dim: int | None = None
        self._count = 0

    def upsert(self, entry: CacheEntry) -> None:
        result = VectorSearchResult(
            cache_key=CacheKey(str(entry.cache_key)),
            embedding=tuple(float(value) for value in entry.embedding),
            score=Score(1.0),
            query=entry.query,
            metadata=entry.metadata,
            payload={"response": entry.response},
        )
        self._add_result(result)

    def _add_result(self, result: VectorSearchResult) -> None:
        """Index a `VectorSearchResult` into the dense matrix + lookup tables.

        Shared by `upsert` (new entries from `CacheEntry`) and
        `FileVectorStore.__init__` (entries decoded from disk), so both paths
        keep the matrix/index structures `search` relies on in sync.
        """

        key = str(result.cache_key)
        embedding = np.asarray(result.embedding, dtype=np.float64)
        if self._dim is None:
            self._dim = int(embedding.shape[0])

        idx = self._key_to_idx.get(key)
        if idx is None:
            idx = self._count
            self._ensure_capacity(self._count + 1)
            self._key_to_idx[key] = idx
            self._idx_to_key.append(key)
            self._namespaces.append(result.metadata.namespace)
            self._count += 1
        else:
            self._namespaces[idx] = result.metadata.namespace

        self._matrix[idx, :] = embedding
        self._norms[idx] = float(np.linalg.norm(embedding))
        self._entries[key] = result

    def delete(self, cache_key: CacheKey) -> None:
        key = str(cache_key)
        idx = self._key_to_idx.pop(key, None)
        if idx is None:
            return
        self._entries.pop(key, None)

        last = self._count - 1
        if idx != last:
            self._matrix[idx, :] = self._matrix[last, :]
            self._norms[idx] = self._norms[last]
            self._namespaces[idx] = self._namespaces[last]
            moved_key = self._idx_to_key[last]
            self._idx_to_key[idx] = moved_key
            self._key_to_idx[moved_key] = idx

        self._idx_to_key.pop()
        self._namespaces.pop()
        self._count -= 1

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        result = self._entries.get(str(cache_key))
        if result is None:
            return None
        return self._copy_result(result)

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
        metadata: CacheMetadata | None = None,
    ) -> list[VectorSearchResult]:
        del metadata
        k = max(0, int(top_k))
        if self._count == 0 or k == 0:
            return []

        query = np.asarray(embedding, dtype=np.float64)
        vectors = self._matrix[: self._count]
        dots = vectors @ query

        if self.similarity == "dot":
            scores = dots
        else:
            query_norm = float(np.linalg.norm(query))
            row_norms = self._norms[: self._count]
            denom = row_norms * query_norm
            scores = np.divide(
                dots, denom, out=np.zeros_like(dots), where=denom > 0.0
            )

        if namespace is not None:
            mask = np.fromiter(
                (ns == namespace for ns in self._namespaces[: self._count]),
                dtype=bool,
                count=self._count,
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(k, self._count)
        if k < self._count:
            top_idx = np.argpartition(-scores, k - 1)[:k]
        else:
            top_idx = np.arange(self._count)

        ordered = sorted(
            top_idx,
            key=lambda i: (-scores[i], self._idx_to_key[i]),
        )

        results: list[VectorSearchResult] = []
        for i in ordered:
            score = float(scores[i])
            if score == -np.inf:
                continue
            entry = self._entries[self._idx_to_key[i]]
            results.append(self._copy_result(entry, score=Score(score)))
        return results

    def clear(self) -> None:
        self._entries.clear()
        self._key_to_idx.clear()
        self._idx_to_key.clear()
        self._namespaces.clear()
        self._matrix = None
        self._norms = None
        self._dim = None
        self._count = 0

    def size(self) -> int:
        return self._count

    def _ensure_capacity(self, needed: int) -> None:
        if self._matrix is None:
            capacity = max(16, needed)
            self._matrix = np.zeros((capacity, self._dim), dtype=np.float64)
            self._norms = np.zeros(capacity, dtype=np.float64)
        elif needed > self._matrix.shape[0]:
            capacity = max(needed, self._matrix.shape[0] * 2)
            new_matrix = np.zeros((capacity, self._dim), dtype=np.float64)
            new_matrix[: self._matrix.shape[0], :] = self._matrix
            self._matrix = new_matrix
            new_norms = np.zeros(capacity, dtype=np.float64)
            new_norms[: self._norms.shape[0]] = self._norms
            self._norms = new_norms

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
