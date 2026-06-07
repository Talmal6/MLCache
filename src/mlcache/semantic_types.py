"""Shared data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, NewType, Sequence

CacheKey = NewType("CacheKey", str)
Query = NewType("Query", str)
Response = NewType("Response", str)
ScorerName = NewType("ScorerName", str)
Score = NewType("Score", float)
Threshold = NewType("Threshold", float)
RegionId = NewType("RegionId", str)
Embedding = Sequence[float]


class OracleDecisionStatus(StrEnum):
    HIT = "hit"
    MISS = "miss"
    ABSTAIN = "abstain"


class TieMode(StrEnum):
    GE = "ge"
    GT = "gt"


class InputSpace(StrEnum):
    EMBEDDING = "embedding"
    SCALAR_SCORE = "scalar_score"
    MIXED = "mixed"
    TEXT_PAIR = "text_pair"


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    namespace: str | None = None
    tenant_id: str | None = None
    model: str | None = None
    region_id: RegionId | None = None
    cluster_id: RegionId | None = None
    created_at: datetime | None = None
    expires_at: datetime | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: CacheKey
    query: Query
    response: Response
    embedding: Embedding
    metadata: CacheMetadata = field(default_factory=CacheMetadata)


@dataclass(frozen=True, slots=True)
class CacheLookup:
    query: Query
    embedding: Embedding
    namespace: str | None = None
    metadata: CacheMetadata = field(default_factory=CacheMetadata)


@dataclass(frozen=True, slots=True)
class OracleDecision:
    status: OracleDecisionStatus
    accepted: bool
    cache_key: CacheKey | None
    score: Score | None
    threshold: Threshold | None
    scorer: ScorerName | None
    candidate_count: int = 0
    accepted_candidate_rank: int | None = None
    reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LabeledPairBatch:
    h0: Sequence[Sequence[float]]
    h1: Sequence[Sequence[float]]
    weights: Sequence[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainCalibEvalSplit:
    h0_train: Sequence[Sequence[float]]
    h1_train: Sequence[Sequence[float]]
    h0_calib: Sequence[Sequence[float]]
    h1_calib: Sequence[Sequence[float]]
    h0_eval: Sequence[Sequence[float]]
    h1_eval: Sequence[Sequence[float]]
    metadata: dict[str, Any] = field(default_factory=dict)
