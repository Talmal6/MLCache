"""External cache boundary contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from oracle import SemanticCacheOracle
from semantic_types import CacheEntry, CacheKey, CacheLookup, OracleDecision, Response


@dataclass(frozen=True, slots=True)
class CacheGatewayResult:
    response: Response | None
    decision: OracleDecision
    metadata: dict[str, Any] = field(default_factory=dict)


class KVStore(ABC):
    """Response storage outside the semantic oracle."""

    @abstractmethod
    def get(self, cache_key: CacheKey) -> Response | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, cache_key: CacheKey, response: Response) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, cache_key: CacheKey) -> None:
        raise NotImplementedError

    @abstractmethod
    def contains(self, cache_key: CacheKey) -> bool:
        raise NotImplementedError


class ExternalSemanticCache(ABC):
    """Public cache facade; semantic correctness is delegated to the oracle."""

    @abstractmethod
    def lookup(self, request: CacheLookup) -> Response | None:
        raise NotImplementedError

    @abstractmethod
    def put(self, entry: CacheEntry) -> CacheKey:
        raise NotImplementedError

    @abstractmethod
    def invalidate(self, cache_key: CacheKey) -> None:
        raise NotImplementedError


class SemanticCacheGateway(ExternalSemanticCache):
    """Public cache gateway coordinating external values and oracle decisions."""

    def __init__(self, *, kv_store: KVStore, oracle: SemanticCacheOracle) -> None:
        self.kv_store = kv_store
        self.oracle = oracle

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        decision = self.oracle.decide(request)
        if not decision.accepted or decision.cache_key is None:
            return CacheGatewayResult(response=None, decision=decision)

        response = self.kv_store.get(decision.cache_key)
        if response is None:
            self.oracle.remove(decision.cache_key)
            return CacheGatewayResult(
                response=None,
                decision=decision,
                metadata={"reason": "kv_key_missing_or_expired"},
            )

        return CacheGatewayResult(response=response, decision=decision)

    def put(self, entry: CacheEntry) -> CacheKey:
        self.kv_store.set(entry.cache_key, entry.response)
        self.oracle.index(entry)
        return entry.cache_key

    def invalidate(self, cache_key: CacheKey) -> None:
        self.oracle.remove(cache_key)
        self.kv_store.delete(cache_key)
