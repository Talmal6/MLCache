"""Semantic cache policy contracts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mlcache.calibration import ThresholdScope
from mlcache.retrieval import VectorSearchResult
from mlcache.semantic_types import CacheEntry, CacheLookup, ScorerName, TieMode


class PolicyAction(StrEnum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, slots=True)
class PolicyContext:
    request: CacheLookup
    candidate: VectorSearchResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    action: PolicyAction
    reason: str | None = None
    scorer: ScorerName | None = None
    threshold_scope: ThresholdScope | None = None
    tie_mode: TieMode | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CachePolicy(ABC):
    """Controls cache admission, reuse eligibility, and oracle routing choices."""

    @abstractmethod
    def can_lookup(self, request: CacheLookup) -> PolicyDecision:
        raise NotImplementedError

    @abstractmethod
    def can_reuse(self, context: PolicyContext) -> PolicyDecision:
        raise NotImplementedError

    @abstractmethod
    def can_store(self, entry: CacheEntry) -> PolicyDecision:
        raise NotImplementedError

    @abstractmethod
    def select_scorer(self, context: PolicyContext) -> ScorerName | None:
        raise NotImplementedError

    @abstractmethod
    def select_threshold_scope(self, context: PolicyContext) -> ThresholdScope:
        raise NotImplementedError
