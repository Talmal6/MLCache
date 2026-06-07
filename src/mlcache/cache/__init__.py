"""Cache gateway and external KV boundaries."""

from mlcache.cache.gateway import CacheGatewayResult, ExternalSemanticCache, SemanticCacheGateway
from mlcache.cache.kv import KVStore

__all__ = [
    "CacheGatewayResult",
    "ExternalSemanticCache",
    "KVStore",
    "SemanticCacheGateway",
]

