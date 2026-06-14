"""FAISS-backed candidate-generation vector store."""

from __future__ import annotations

import hashlib
import os
import threading
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from mlcache.retrieval.vector_store import VectorSearchResult, VectorStore
from mlcache.semantic_types import CacheEntry, CacheKey, CacheMetadata, Embedding, Score

if TYPE_CHECKING:
    from mlcache.persistence.postgres import PostgresCacheRepository


class FaissVectorStore(VectorStore):
    """FAISS vector index with optional PostgreSQL candidate validation.

    FAISS stores only vectors and integer IDs. When `repository` is supplied,
    search results are resolved through PostgreSQL and only ACTIVE, non-expired
    rows matching the request scope are returned.
    """

    def __init__(
        self,
        *,
        dim: int | None = None,
        index_path: str | Path | None = None,
        repository: PostgresCacheRepository | None = None,
        metric: str = "cosine",
        shard_id: str = "default",
        overfetch_factor: int = 5,
        autoload: bool = True,
    ) -> None:
        if metric != "cosine":
            raise ValueError("FaissVectorStore currently supports metric='cosine'")
        if dim is not None and int(dim) <= 0:
            raise ValueError("dim must be positive")
        if int(overfetch_factor) <= 0:
            raise ValueError("overfetch_factor must be positive")
        self.dim = int(dim) if dim is not None else None
        self.index_path = Path(index_path) if index_path is not None else None
        self.repository = repository
        self.metric = metric
        self.shard_id = shard_id
        self.overfetch_factor = int(overfetch_factor)
        self._lock = threading.RLock()
        self._index: Any | None = None
        self._local_entries: dict[int, VectorSearchResult] = {}
        self._local_key_to_id: dict[str, int] = {}
        if self.dim is not None:
            self._index = self._new_index(self.dim)
        if autoload and self.index_path is not None and self.index_path.exists():
            self.load_snapshot(self.index_path)

    def upsert(self, entry: CacheEntry) -> None:
        faiss_id = _faiss_id_for_entry(entry)
        result = VectorSearchResult(
            cache_key=CacheKey(str(entry.cache_key)),
            embedding=tuple(float(value) for value in entry.embedding),
            score=Score(1.0),
            query=entry.query,
            metadata=entry.metadata,
            payload={"response": entry.response, "faiss_id": faiss_id},
        )
        self.add_with_id(faiss_id, entry.embedding, result=result)

    def add_with_id(
        self,
        faiss_id: int,
        embedding: Embedding,
        *,
        result: VectorSearchResult | None = None,
    ) -> None:
        vector = _normalize_vector(embedding)
        with self._lock:
            self._ensure_index(vector.shape[0])
            self._remove_id_locked(int(faiss_id))
            self._index.add_with_ids(vector.reshape(1, -1), np.asarray([int(faiss_id)], dtype=np.int64))
            if result is not None:
                self._local_entries[int(faiss_id)] = _copy_result(
                    result,
                    score=Score(1.0),
                    payload={**result.payload, "faiss_id": int(faiss_id)},
                )
                self._local_key_to_id[str(result.cache_key)] = int(faiss_id)

    def delete(self, cache_key: CacheKey) -> None:
        with self._lock:
            faiss_id = self._local_key_to_id.pop(str(cache_key), None)
            if faiss_id is None:
                return
            self._local_entries.pop(faiss_id, None)
            self._remove_id_locked(faiss_id)

    def get(self, cache_key: CacheKey) -> VectorSearchResult | None:
        if self.repository is not None:
            return self.repository.get_vector_result(cache_key, require_active=True)
        with self._lock:
            faiss_id = self._local_key_to_id.get(str(cache_key))
            if faiss_id is None:
                return None
            result = self._local_entries.get(faiss_id)
        return None if result is None else _copy_result(result)

    def search(
        self,
        embedding: Embedding,
        *,
        namespace: str | None = None,
        top_k: int = 10,
        metadata: CacheMetadata | None = None,
    ) -> list[VectorSearchResult]:
        k = max(0, int(top_k))
        if k == 0:
            return []
        query = _normalize_vector(embedding)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            if int(query.shape[0]) != int(self.dim):
                raise ValueError(f"embedding dimension {query.shape[0]} does not match FAISS dimension {self.dim}")
            search_k = min(int(self._index.ntotal), max(k, k * self.overfetch_factor))
            distances, ids = self._index.search(query.reshape(1, -1), search_k)
        faiss_scores = [
            (int(faiss_id), float(score))
            for faiss_id, score in zip(ids[0], distances[0], strict=False)
            if int(faiss_id) >= 0
        ]
        if self.repository is not None:
            return self.repository.fetch_active_results_by_faiss_ids(
                faiss_scores,
                namespace=namespace,
                metadata=metadata,
            )[:k]
        return self._local_results(faiss_scores, namespace=namespace)[:k]

    def save_snapshot(self, path: str | Path | None = None) -> dict[str, Any]:
        target = Path(path) if path is not None else self.index_path
        if target is None:
            raise ValueError("snapshot path is required")
        with self._lock:
            if self._index is None:
                raise ValueError("cannot snapshot an uninitialized FAISS index")
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.tmp")
            _faiss().write_index(self._index, str(tmp))
            os.replace(tmp, target)
        checksum = checksum_file(target)
        return {
            "snapshot_uri": str(target),
            "checksum": checksum,
            "index_type": "IndexIDMap2(IndexFlatIP)",
            "shard_id": self.shard_id,
        }

    def load_snapshot(self, path: str | Path, *, expected_checksum: str | None = None) -> None:
        target = Path(path)
        if expected_checksum is not None:
            actual = checksum_file(target)
            if actual != expected_checksum:
                raise ValueError(f"FAISS snapshot checksum mismatch for {target}")
        index = _faiss().read_index(str(target))
        with self._lock:
            self._index = index
            self.dim = int(index.d)

    def size(self) -> int:
        with self._lock:
            return 0 if self._index is None else int(self._index.ntotal)

    def _ensure_index(self, dim: int) -> None:
        if self._index is None:
            self.dim = int(dim)
            self._index = self._new_index(self.dim)
            return
        if int(dim) != int(self.dim):
            raise ValueError(f"embedding dimension {dim} does not match FAISS dimension {self.dim}")

    def _remove_id_locked(self, faiss_id: int) -> None:
        if self._index is None or self._index.ntotal == 0:
            return
        selector = np.asarray([int(faiss_id)], dtype=np.int64)
        try:
            self._index.remove_ids(selector)
        except Exception:
            return

    def _local_results(
        self,
        faiss_scores: list[tuple[int, float]],
        *,
        namespace: str | None,
    ) -> list[VectorSearchResult]:
        results: list[VectorSearchResult] = []
        with self._lock:
            for faiss_id, score in faiss_scores:
                result = self._local_entries.get(int(faiss_id))
                if result is None:
                    continue
                if namespace is not None and result.metadata.namespace != namespace:
                    continue
                results.append(_copy_result(result, score=Score(float(score))))
        return results

    @staticmethod
    def _new_index(dim: int) -> Any:
        faiss = _faiss()
        return faiss.IndexIDMap2(faiss.IndexFlatIP(int(dim)))


class FaissOutboxIndexer:
    """Consumes PostgreSQL outbox events and applies them to FAISS."""

    def __init__(self, *, repository: PostgresCacheRepository, vector_store: FaissVectorStore) -> None:
        self.repository = repository
        self.vector_store = vector_store

    def process_once(self, *, limit: int = 100) -> int:
        processed = 0
        for event in self.repository.pending_outbox_events(limit=limit):
            if event.event_type == "UPSERT_ENTRY" and event.faiss_id is not None:
                result = self.repository.get_indexable_entry_by_faiss_id(event.faiss_id)
                if result is None:
                    continue
                self.vector_store.add_with_id(event.faiss_id, result.embedding, result=result)
                self.repository.mark_entry_indexed(faiss_id=event.faiss_id)
                self.repository.mark_outbox_processed(event.event_id)
                processed += 1
            elif event.event_type == "DELETE_ENTRY":
                if event.cache_key is not None:
                    self.vector_store.delete(event.cache_key)
                self.repository.mark_outbox_processed(event.event_id)
                processed += 1
        return processed


def checksum_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("FaissVectorStore requires: pip install -e '.[faiss]'") from exc
    return faiss


def _normalize_vector(values: Embedding) -> np.ndarray:
    vector = np.asarray(tuple(float(value) for value in values), dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("embedding must be one-dimensional")
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return np.ascontiguousarray(vector, dtype=np.float32)


def _faiss_id_for_entry(entry: CacheEntry) -> int:
    raw = entry.metadata.attributes.get("faiss_id")
    if raw is not None:
        return int(raw)
    try:
        value = uuid.UUID(str(entry.cache_key)).int & ((1 << 63) - 1)
        return int(value)
    except ValueError:
        digest = hashlib.blake2b(str(entry.cache_key).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big", signed=False) & ((1 << 63) - 1)


def _copy_result(
    result: VectorSearchResult,
    *,
    score: Score | None = None,
    payload: dict[str, Any] | None = None,
) -> VectorSearchResult:
    return VectorSearchResult(
        cache_key=CacheKey(str(result.cache_key)),
        embedding=tuple(float(value) for value in result.embedding),
        score=score if score is not None else Score(float(result.score)),
        query=result.query,
        metadata=result.metadata,
        payload=dict(result.payload if payload is None else payload),
    )


__all__ = ["FaissOutboxIndexer", "FaissVectorStore", "checksum_file"]
