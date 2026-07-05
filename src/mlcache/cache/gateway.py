"""External cache boundary contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from mlcache.cache.eviction import CacheEvictionPolicy
from mlcache.cache.kv import KVStore
from mlcache.oracle.base import SemanticCacheOracle
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, OracleDecision, Response


@dataclass(frozen=True, slots=True)
class CacheGatewayResult:
    response: Response | None
    decision: OracleDecision
    metadata: dict[str, Any] = field(default_factory=dict)


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
    """Public cache gateway coordinating external values and oracle decisions.

    When an ``eviction_policy`` and a positive ``max_entries`` are configured,
    the gateway enforces the capacity bound: every insert past the limit evicts
    the victim chosen by the policy (LRU/LFU/FIFO). Eviction is independent of
    the oracle's semantic HIT/MISS decision — it only governs which already
    admitted entry is dropped to make room.
    """

    def __init__(
        self,
        *,
        kv_store: KVStore,
        oracle: SemanticCacheOracle,
        eviction_policy: CacheEvictionPolicy | None = None,
        max_entries: int | None = None,
    ) -> None:
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be a positive integer or None")
        self.kv_store = kv_store
        self.oracle = oracle
        self.eviction_policy = eviction_policy
        self.max_entries = max_entries

    def lookup(self, request: CacheLookup) -> Response | None:
        return self.lookup_with_decision(request).response

    def lookup_with_decision(self, request: CacheLookup) -> CacheGatewayResult:
        decision = self.oracle.decide(request)
        if not decision.accepted or decision.cache_key is None:
            return CacheGatewayResult(response=None, decision=decision)

        response = self.kv_store.get(decision.cache_key)
        if response is None:
            self._record_removal(decision.cache_key)
            self.oracle.remove(decision.cache_key)
            return CacheGatewayResult(
                response=None,
                decision=decision,
                metadata={"reason": "kv_key_missing_or_expired"},
            )

        # Cache HIT: refresh recency/frequency bookkeeping for eviction.
        if self.eviction_policy is not None:
            self.eviction_policy.record_hit(decision.cache_key)
        return CacheGatewayResult(response=response, decision=decision)

    def put(self, entry: CacheEntry) -> CacheKey:
        self.kv_store.set(entry.cache_key, entry.response)
        self.oracle.index(entry)
        if self.eviction_policy is not None:
            self.eviction_policy.record_insert(entry.cache_key)
            self._enforce_capacity()
        return entry.cache_key

    def invalidate(self, cache_key: CacheKey) -> None:
        self._record_removal(cache_key)
        self.oracle.remove(cache_key)
        self.kv_store.delete(cache_key)

    def _enforce_capacity(self) -> None:
        policy = self.eviction_policy
        if policy is None or self.max_entries is None:
            return
        while len(policy) > self.max_entries:
            victim = policy.select_victim()
            if victim is None:
                break
            self.invalidate(victim)

    def _record_removal(self, cache_key: CacheKey) -> None:
        if self.eviction_policy is not None:
            self.eviction_policy.record_removal(cache_key)
