"""Semantic cache oracle contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from mlcache.online import OnlineBatch, StopStatus
from mlcache.semantic_types import CacheEntry, CacheKey, CacheLookup, OracleDecision, TrainCalibEvalSplit


class SemanticCacheOracle(ABC):
    """Owns all semantic correctness logic for cache reuse."""

    @abstractmethod
    def decide(self, request: CacheLookup) -> OracleDecision:
        raise NotImplementedError

    @abstractmethod
    def fit(self, split: TrainCalibEvalSplit) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_online(self, batch: OnlineBatch) -> StopStatus:
        raise NotImplementedError

    @abstractmethod
    def index(self, entry: CacheEntry) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove(self, cache_key: CacheKey) -> None:
        raise NotImplementedError

