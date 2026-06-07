"""Judge data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mlcache.semantic_types import CacheKey, CacheMetadata, Query, Response, Score


class JudgeLabel(StrEnum):
    REUSABLE = "reusable"
    NOT_REUSABLE = "not_reusable"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    query: Query
    candidate_query: Query | None = None
    candidate_response: Response | None = None
    candidate_key: CacheKey | None = None
    metadata: CacheMetadata = field(default_factory=CacheMetadata)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    label: JudgeLabel
    confidence: Score | None = None
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.label == JudgeLabel.REUSABLE


@dataclass(frozen=True, slots=True)
class JudgeResult:
    request: JudgeRequest
    decision: JudgeDecision
