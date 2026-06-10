"""File-backed local vector store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlcache.persistence import atomic_write_json, decode_cache_metadata, encode_cache_metadata, json_safe, read_json_or_default
from mlcache.retrieval.in_memory import InMemoryVectorStore
from mlcache.retrieval.vector_store import VectorSearchResult
from mlcache.semantic_types import CacheEntry, CacheKey, Query, Score


class FileVectorStore(InMemoryVectorStore):
    """JSON-backed vector store for restart-safe local experiments."""

    def __init__(self, path: str | Path, *, similarity: str = "cosine") -> None:
        self.path = Path(path)
        super().__init__(similarity=similarity)
        data = read_json_or_default(self.path, {"records": [], "similarity": similarity})
        self.similarity = str(data.get("similarity") or similarity)
        if self.similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")
        for item in data.get("records", ()):
            result = self._decode_result(item)
            self._add_result(result)

    def upsert(self, entry: CacheEntry) -> None:
        super().upsert(entry)
        self._persist()

    def delete(self, cache_key: CacheKey) -> None:
        super().delete(cache_key)
        self._persist()

    def clear(self) -> None:
        super().clear()
        self._persist()

    def _persist(self) -> None:
        atomic_write_json(
            self.path,
            {
                "format": "mlcache.file_vector_store.v1",
                "similarity": self.similarity,
                "records": [
                    self._encode_result(self._entries[key])
                    for key in sorted(self._entries, key=str)
                ],
            },
        )

    @staticmethod
    def _encode_result(result: VectorSearchResult) -> dict[str, Any]:
        return {
            "cache_key": str(result.cache_key),
            "query": None if result.query is None else str(result.query),
            "embedding": [float(value) for value in result.embedding],
            "metadata": encode_cache_metadata(result.metadata),
            "payload": json_safe(result.payload),
        }

    @staticmethod
    def _decode_result(data: dict[str, Any]) -> VectorSearchResult:
        return VectorSearchResult(
            cache_key=CacheKey(str(data["cache_key"])),
            embedding=tuple(float(value) for value in data.get("embedding", ())),
            score=Score(1.0),
            query=Query(data["query"]) if data.get("query") is not None else None,
            metadata=decode_cache_metadata(data.get("metadata")),
            payload=dict(data.get("payload") or {}),
        )


__all__ = ["FileVectorStore"]
