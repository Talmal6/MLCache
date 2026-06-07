"""Compatibility wrapper for the old flat cache module."""

from mlcache.cache import (
    CacheGatewayResult,
    ExternalSemanticCache,
    FileKVStore,
    InMemoryKVStore,
    KVStore,
    SemanticCacheGateway,
)

__all__ = [
    "CacheGatewayResult",
    "ExternalSemanticCache",
    "FileKVStore",
    "InMemoryKVStore",
    "KVStore",
    "SemanticCacheGateway",
]
