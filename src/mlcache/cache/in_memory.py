"""In-memory cache backends."""

from __future__ import annotations

from mlcache.cache.kv import KVStore
from mlcache.semantic_types import CacheKey, Response


class InMemoryKVStore(KVStore):
    """Deterministic in-memory response store."""

    def __init__(self, initial: dict[CacheKey | str, Response | str] | None = None) -> None:
        self._responses: dict[CacheKey, Response] = {}
        for key, value in (initial or {}).items():
            self.set(CacheKey(str(key)), Response(str(value)))

    def get(self, cache_key: CacheKey) -> Response | None:
        return self._responses.get(cache_key)

    def set(self, cache_key: CacheKey, response: Response) -> None:
        self._responses[CacheKey(str(cache_key))] = Response(str(response))

    def delete(self, cache_key: CacheKey) -> None:
        self._responses.pop(CacheKey(str(cache_key)), None)

    def contains(self, cache_key: CacheKey) -> bool:
        return CacheKey(str(cache_key)) in self._responses

    def clear(self) -> None:
        self._responses.clear()

    def size(self) -> int:
        return len(self._responses)


__all__ = ["InMemoryKVStore"]
