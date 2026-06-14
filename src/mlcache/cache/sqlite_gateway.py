"""SQLite-first semantic cache gateway."""

from __future__ import annotations

from time import perf_counter

from mlcache.cache.gateway import CacheGatewayResult, SemanticCacheGateway
from mlcache.cache.kv import KVStore
from mlcache.oracle.base import SemanticCacheOracle
from mlcache.persistence.sqlite import SQLiteCacheRepository
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, Response


class SQLiteSemanticCacheGateway(SemanticCacheGateway):
    """Gateway that makes SQLite the source of truth for writes and deletes."""

    def __init__(
        self,
        *,
        kv_store: KVStore,
        oracle: SemanticCacheOracle,
        repository: SQLiteCacheRepository,
    ) -> None:
        super().__init__(kv_store=kv_store, oracle=oracle)
        self.repository = repository

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        started = perf_counter()
        result = super().lookup_with_decision(request)
        latency_ms = int((perf_counter() - started) * 1000)
        self.repository.record_query_event(request, result.decision, latency_ms=latency_ms)
        return result

    def put(self, entry: CacheEntry) -> CacheKey:
        self.repository.insert_pending_entry(entry)
        return entry.cache_key

    def invalidate(self, cache_key: CacheKey) -> None:
        self.repository.tombstone_entry(cache_key)
        self.oracle.remove(cache_key)


__all__ = ["SQLiteSemanticCacheGateway"]
