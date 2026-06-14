"""Compatibility wrapper for the old flat cache module."""

from mlcache.cache import (
    CacheGatewayResult,
    ExternalSemanticCache,
    FileKVStore,
    InMemoryKVStore,
    KVStore,
    MySQLKVStore,
    MySQLSemanticCacheGateway,
    PostgresKVStore,
    PostgresSemanticCacheGateway,
    SemanticCacheGateway,
)

__all__ = [
    "CacheGatewayResult",
    "ExternalSemanticCache",
    "FileKVStore",
    "InMemoryKVStore",
    "KVStore",
    "MySQLKVStore",
    "MySQLSemanticCacheGateway",
    "PostgresKVStore",
    "PostgresSemanticCacheGateway",
    "SemanticCacheGateway",
]
